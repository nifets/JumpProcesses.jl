"""
Tau splitting, implementation following:
"An exact tau-leaping method", Ron Solan and Gad Getz, 2025
"""

struct ReactionEntry{T}
    idx::Int      # i
    count::Int    # eᵢ
    rate_low::T   # lᵢ
    rate_high::T  # uᵢ
end

@inline max_rand(::Type{T}, rng, n) where {T} = n == 0 ? zero(T) : rand(rng)^inv(n)
@inline min_rand(::Type{T}, rng, n) where {T} = n == 0 ? one(T) : -expm1(log(rand(rng)) / n)

function ReactionEntry(idx, rate::T, Δt, rng::Random.AbstractRNG) where {T}
    rate_high = rate + randexp(rng, T) / Δt
    iszero(rate) && return ReactionEntry(idx, 0, zero(T), rate_high)
    count = pois_rand(rng, Δt*rate)
    rate_low = rate * max_rand(T, rng, count)
    ReactionEntry(idx, count, rate_low, rate_high)
end

function split(rx::ReactionEntry{T}, Δt, rng) where {T}
    lcount = binom_rand(rng, rx.count, 0.5)
    rcount = rx.count - lcount

    lhigh, rhigh = rx.rate_high, rx.rate_high + randexp(rng, T) / Δt * 2
    rand(rng, Bool) && ((lhigh, rhigh) = (rhigh, lhigh))

    if rx.count == 0
        llow = zero(T)
        rlow = zero(T)
    elseif rand(rng) <= lcount / rx.count
        llow = rx.rate_low
        rlow = rx.rate_low * max_rand(T, rng, rcount)
    else
        rlow = rx.rate_low
        llow = rx.rate_low * max_rand(T, rng, lcount)
    end

    return ReactionEntry(rx.idx, lcount, llow, lhigh),
    ReactionEntry(rx.idx, rcount, rlow, rhigh)
end

"""
resample reaction precursor counts and rate bounds according to new rate `rate`
"""
function resample(rx::ReactionEntry{T}, rate::T, Δt, rng) where {T}
    if rate >= rx.rate_high
        Δr = rate - rx.rate_high
        new_pts = pois_rand(rng, Δt*Δr)
        e = rx.count + 1 + new_pts
        lb = rx.rate_high + Δr * max_rand(T, rng, new_pts)
        ub = rate + randexp(rng, T) / Δt
    elseif rate >= rx.rate_low
        return rx
    else
        e = binom_rand(rng, rx.count - 1, rate / rx.rate_low)
        lb = rate * max_rand(T, rng, e)
        ub = rate + (rx.rate_low - rate) * min_rand(T, rng, rx.count - e - 1)
    end

    ReactionEntry(rx.idx, e, lb, ub)
end

mutable struct TauSplittingNode{T}
    t0::T
    Δt::T
    is_left::Bool
    is_precursor::Bool
    active::Vector{ReactionEntry{T}}
    const inactive::Vector{ReactionEntry{T}}
    pending::Vector{ReactionEntry{T}} # hold the right sibling entries when processing left node
end

"""
Turn a left sibling node into a right sibling node due for processing
"""
@inline function advance_right!(node::TauSplittingNode)
    @assert node.is_left "advance_right! should only be called on left nodes"
    node.active, node.pending = node.pending, node.active
    node.t0 += node.Δt
    node.is_left = false
    node.is_precursor = true
    nothing
end

mutable struct TauSplittingJumpAggregation{T, S, F1, F2, RNG, DEPGR, VJMAP, PB, U} <:
               AbstractSSAJumpAggregator{T, S, F1, F2, RNG}
    next_jump::Int     # not used
    prev_jump::Int     # not used
    next_jump_time::T
    end_time::T
    cur_rates::Nothing # not used
    sum_rate::Nothing  # not used
    ma_jumps::S
    rates::F1
    affects!::F2       # not used
    save_positions::Tuple{Bool, Bool}
    rng::RNG
    dep_gr::DEPGR
    jumptostoich_map::Vector{Vector{Pair{Int, Int}}}
    vartojumps_map::VJMAP
    spec_to_writer_rxs::Vector{Vector{Int}}
    rx_to_reader_specs::Vector{Vector{Int}}
    propensity_bounds::PB
    prev_jump_time::T
    ulow::U
    uhigh::U
    ulow_rx::U
    nodes::Vector{TauSplittingNode{T}}
    num_unstable_by_spec::Vector{Int} # how many reactions are unstable for each reactant
    rx_inactive_node::Vector{Int}     # 0 if active
    rx_stable::Vector{Bool}
    max_interval::T
end

function TauSplittingJumpAggregation(nj::Int, njt::T, et::T, crs::Nothing, sr::Nothing,
    maj::S, rs::F1, affs!::F2, sps::Tuple{Bool, Bool}, rng::RNG;
    u::U, dep_graph = nothing, vartojumps_map = nothing, jumptovars_map = nothing, jumptostoich_map = nothing, max_interval = typemax(T), propensity_bounds=IncreasingBounds(), kwargs...) where {T,S,F1,F2,RNG,U}

    numspec = length(u)
    numrxs = get_num_majumps(maj) + length(rs)

    if dep_graph === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use ConstantRateJumps with the TauSplitting algorithm a dependency graph must be supplied.")
        else
            dg = make_dependency_graph(length(u), maj)
        end
    else
        dg = dep_graph
    end

    stochtype = Vector{Vector{Pair{Int, Int}}}
    if jumptostoich_map === nothing
        isempty(rs) ||
            error("To use ConstantRateJumps with the TauSplitting algorithm a map from jumps to their net stoichiometry must be supplied (via `jumptostoich_map`).")
        jtos_map = stochtype()
    else
        length(jumptostoich_map) == length(rs) ||
            error("`jumptostoich_map` must have one entry per ConstantRateJump (got $(length(jumptostoich_map)) for $(length(rs)) jumps.")
        jtos_map = convert(stochtype, jumptostoich_map)
    end

    if vartojumps_map === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use the Tau Splitting algorithm a map from variables to dependent jumps must be supplied.")
        else
            vtoj_map = var_to_jumps_map(numspec, maj)
        end
    else
        vtoj_map = vartojumps_map
    end

    if jumptovars_map === nothing
        if (get_num_majumps(maj) == 0) || !isempty(rs)
            error("To use the Tau Splitting algorithm a map from jumps to dependent variables must be supplied.")
        else
            jtov_map = jump_to_vars_map(maj)
        end
    else
        jtov_map = jumptovars_map
    end

    # invert jumptovars_map: for each species, the reaction that can change it
    spec_to_writer_rxs = [Int[] for _ in 1:numspec]
    for (rx, specs) in pairs(jtov_map), spec in specs
        push!(spec_to_writer_rxs[spec], rx)
    end

    rx_to_reader_specs = [Int[] for _ in 1:numrxs]
    for (spec, rxs) in pairs(vtoj_map), rx in rxs
        push!(rx_to_reader_specs[rx], spec)
    end

    pb = propensity_bounds

    affecttype = F2 <: Tuple ? F2 : Any
    TauSplittingJumpAggregation{T, S, F1, affecttype, RNG, typeof(dg), typeof(vtoj_map), typeof(pb), U}(nj, nj, njt, et, crs, sr, maj, rs, affs!, sps, rng, dg, jtos_map, vtoj_map, spec_to_writer_rxs, rx_to_reader_specs, pb, njt, similar(u), similar(u), copy(u), TauSplittingNode{T}[], zeros(Int, numspec), zeros(Int, numrxs), trues(numrxs), convert(T, max_interval))
end

function aggregate(aggregator::TauSplitting, u, p, t, end_time, constant_jumps,
        ma_jumps, save_positions, rng; kwargs...)
    rates, affects! = get_jump_info_fwrappers(u, p, t, constant_jumps)
    next_jump = 0
    next_jump_time = typemax(t)
    TauSplittingJumpAggregation(next_jump, next_jump_time, end_time, nothing, nothing,
        ma_jumps, rates, affects!, save_positions, rng; u, kwargs...)
end

function initialize!(p::TauSplittingJumpAggregation, integrator, u, params, t)
    p.end_time = integrator.sol.prob.tspan[2]
    p.prev_jump_time = t
    generate_jumps!(p, integrator, u, params, t)
    nothing
end

function execute_jumps!(p::TauSplittingJumpAggregation, integrator, u, params, t, affects!)
    advance_to!(integrator, p, t)
    nothing
end

function advance_to!(integrator, p::TauSplittingJumpAggregation, t)
    u = integrator.u
    params = integrator.p
    t0 = p.prev_jump_time
    Δt = t - t0
    Δt > zero(Δt) || return nothing

    copyto!(p.ulow, u)
    copyto!(p.uhigh, u)
    fill!(p.rx_stable, true)
    fill!(p.num_unstable_by_spec, 0)
    node = init_node!(p, 1, t0, Δt)

    for rxidx in 1:num_rxs(p)
        rate = jump_rate(p, rxidx, u, params, t0)
        push!(node.active, ReactionEntry(rxidx, rate, Δt, p.rng))
    end
    process_node!(p, 1, integrator, params)
    p.prev_jump_time = t

    nothing
end

function generate_jumps!(p::TauSplittingJumpAggregation, integrator, u, params, t)
    p.next_jump_time = min(p.end_time, t + p.max_interval)
    nothing
end

######################## τ-splitting specific helper routines #########################

# the main algorithm that recursively processes a node over [t0, t0+Δt]
function process_node!(p::TauSplittingJumpAggregation, depth, integrator, params)
    node = p.nodes[depth]
    active = node.active

    node.is_precursor && resample!(p, depth, integrator, params)

    # update ulow and uhigh to reflect resampled reaction counts
    for rx in active
        add_slack!(p, rx)
    end

    # update reaction stability and dependents
    i = 1
    while i <= length(active)
        rx = active[i]
        stable_before = p.rx_stable[rx.idx]
        stable_now = is_stable(p, rx, params, node.t0)
        if stable_before != stable_now
            p.rx_stable[rx.idx] = stable_now
            for spec in jump_inputs(p, rx.idx)
                if stable_now
                    p.num_unstable_by_spec[spec] -= 1
                else
                    if p.num_unstable_by_spec[spec] == 0
                        for rxj in p.spec_to_writer_rxs[spec]
                            finalrx = reactivate!(p, rxj, depth, integrator)
                            finalrx === nothing || add_slack!(p, finalrx)
                        end
                    end
                    p.num_unstable_by_spec[spec] += 1
                end
            end
        end
        i += 1
    end

    # deactivate all the reactions that we can
    keep = 0
    for rx in active
        if can_deactivate(p, rx)
            p.rx_inactive_node[rx.idx] = depth
            push!(node.inactive, rx)
        else
            keep += 1
            active[keep] = rx
        end
    end
    resize!(active, keep)

    # for reactions that remain active, process them recursively
    if !isempty(active)
        child = init_node!(p, depth+1, node.t0, node.Δt / 2)
        while !isempty(active)
            rx = pop!(active)
            remove_slack!(p, rx)
            left, right = split(rx, node.Δt, p.rng)
            push!(child.active, left)
            push!(child.pending, right)
        end
        process_node!(p, depth + 1, integrator, params)
        advance_right!(child)
        process_node!(p, depth + 1, integrator, params)
    end

    # remaining reactions can now safely be added to the state
    # so that integrator.u reflects the true state at time t1
    while !isempty(node.inactive)
        rx = pop!(node.inactive)
        add_net_stoch!(p, integrator.u, rx)
        remove_slack!(p, rx)
        add_net_stoch!(p, p.ulow, rx)
        add_net_stoch!(p, p.uhigh, rx)
        p.rx_inactive_node[rx.idx] = 0
    end

    nothing
end

function resample!(p::TauSplittingJumpAggregation, depth::Int, integrator, params)
    node = p.nodes[depth]
    @assert node.is_precursor "resample! should only be called on precursor nodes"

    active = node.active
    i = 1
    while i <= length(active)
        rxp = active[i] # precursor reaction
        rate = jump_rate(p, rxp.idx, integrator.u, params, node.t0)
        rx = resample(rxp, rate, node.Δt, p.rng)
        active[i] = rx
        # if a reaction has more firings after resampling, reactivate its inactive dependents
        if rx.count > rxp.count
            for dep in p.dep_gr[rx.idx]
                reactivate!(p, dep, depth, integrator)
            end
        end
        i += 1
    end

    node.is_precursor = false
    nothing
end

function reactivate!(p::TauSplittingJumpAggregation, rxidx, curr_depth, integrator)
    depth = p.rx_inactive_node[rxidx]
    depth == 0 && return

    node = p.nodes[depth]

    i = findfirst(r -> r.idx == rxidx, node.inactive)
    rx = node.inactive[i]
    deleteat!(node.inactive, i)
    p.rx_inactive_node[rxidx] = 0

    remove_slack!(p, rx)

    Δt = node.Δt
    for d in (depth + 1):curr_depth
        node = p.nodes[d]
        left, right = split(rx, Δt, p.rng)
        if node.is_left
            push!(node.pending, right)
            rx = left
        else
            add_net_stoch!(p, integrator.u, left)
            add_net_stoch!(p, p.ulow, left)
            add_net_stoch!(p, p.uhigh, left)
            rx = right
        end
        Δt = node.Δt
    end

    push!(p.nodes[curr_depth].active, rx)

    rx
end

function init_node!(p::TauSplittingJumpAggregation{T}, depth, t0, Δt) where {T}
    while (depth > length(p.nodes))
        push!(p.nodes,
            TauSplittingNode(zero(T), zero(T), true, false, ReactionEntry{T}[],
                ReactionEntry{T}[], ReactionEntry{T}[]))
    end
    node = p.nodes[depth]
    empty!(node.active);
    empty!(node.inactive);
    empty!(node.pending)
    node.t0 = t0
    node.Δt = Δt
    node.is_left = true
    node.is_precursor=false
    node
end

@inline function net_stoch(p, rxidx)
    nummaj = get_num_majumps(p.ma_jumps)
    @inbounds rxidx <= nummaj ? p.ma_jumps.net_stoch[rxidx] : p.jumptostoich_map[rxidx - nummaj]
end

@inline jump_inputs(p, rxidx) = p.rx_to_reader_specs[rxidx]
@inline jump_outputs(p, rxidx) = (spec for (spec, _) in net_stoch(p, rxidx))

@inline num_rxs(p) = get_num_majumps(p.ma_jumps) + length(p.rates)

@inline function jump_rate(p, rxidx, u, params, t)
    calculate_jump_rate(
        p.ma_jumps, get_num_majumps(p.ma_jumps), p.rates, u, params, t, rxidx)
end

@inline function add_net_stoch!(p, dest, rx)
    @inbounds for (spec, stoch) in net_stoch(p, rx.idx)
        dest[spec] += rx.count * stoch
    end
    nothing
end

@inline function change_slack!(p::TauSplittingJumpAggregation, rx::ReactionEntry, sign)
    (; ulow, uhigh) = p
    e = rx.count
    @inbounds for (spec, stoch) in net_stoch(p, rx.idx)
        if stoch > 0
            uhigh[spec] += e * stoch * sign
        else
            ulow[spec] += e * stoch * sign
        end
    end
    nothing
end

@inline add_slack!(p, rx) = change_slack!(p, rx, 1)
@inline remove_slack!(p, rx) = change_slack!(p, rx, -1)


"""
compute a reaction specific lower bound that is tighter than `p.ulow`
"""
@inline function lower_state!(p::TauSplittingJumpAggregation, rx::ReactionEntry)
    v = p.ulow_rx
    neg = false
    @inbounds for spec in jump_inputs(p, rx.idx)
        x = p.ulow[spec]
        x < 0 && (neg = true)
        v[spec] = x
    end
    δ = min(rx.count, 1)
    @inbounds for (spec, stoch) in net_stoch(p, rx.idx)
        stoch < 0 || continue
        x = p.ulow[spec] - δ * stoch
        x < 0 && (neg = true)
        v[spec] = x
    end
    return neg
end

@inline function is_stable(p::TauSplittingJumpAggregation, rx::ReactionEntry, params, t)
    # C1
    jump_upper_bound(p.propensity_bounds, rx.idx, p.ulow, p.uhigh, p.ma_jumps, p.rates, params, t) < rx.rate_high || return false
    # C2
    lower_state!(p, rx) && return false
    return rx.rate_low <= jump_lower_bound(p.propensity_bounds, rx.idx, p.ulow, p.uhigh, p.ma_jumps, p.rates, params, t)
end

# because an inactive reaction's count is not accounted for in the state `u` during a node's processing, we must ensure its dependents are stable:
# unstable reactions incur a reevaluation of the rate against the true state when they get resampled in a deeper node
@inline function can_deactivate(p::TauSplittingJumpAggregation, rx::ReactionEntry)
    p.rx_stable[rx.idx] || return false
    rx.count == 0 && return true
    for spec in jump_outputs(p, rx.idx)
        p.num_unstable_by_spec[spec] == 0 || return false
    end
    return true
end


# Binomial random variate generation, implementation following:
# W. Hörmann, "The generation of binomial random variates",
# Journal of Statistical Computation and Simulation, 46, pg. 101-110 (1993).

# tail term of the Stirling series, log(k!) - log(sqrt(2πk) (k/e)^k)
@inline function stirling_tail(k::Int)
    tbl = (0.0810614667953272, 0.0413406959554093, 0.0276779256849983,
        0.0207906721037650, 0.0166446911898211, 0.0138761288230707,
        0.0118967099458917, 0.0104112652619720, 0.00925546218271273,
        0.00833056343336287)
    k < 10 && return @inbounds tbl[k + 1]
    kp1sq = (k + 1.0)^2
    return (1 / 12 - (1 / 360 - 1 / 1260 / kp1sq) / kp1sq) / (k + 1)
end

# inversion, O(np) expected — only used for small np
function binom_rand_inversion(rng, n::Int, p::Float64)
    q = 1 - p
    s = p / q
    a = (n + 1) * s
    r = q^n
    u = rand(rng)
    x = 0
    while u > r
        u -= r
        x += 1
        x > n && return n
        r *= a / x - s
    end
    return x
end

# transformed rejection, O(1)
function binom_rand_btrs(rng, n::Int, p::Float64)
    q = 1 - p
    spq = sqrt(n * p * q)
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * p
    c = n * p + 0.5
    v_r = 0.92 - 4.2 / b
    r = p / q
    alpha = (2.83 + 5.1 / b) * spq
    m = floor(Int, (n + 1) * p)

    while true
        u = rand(rng) - 0.5
        v = rand(rng)
        us = 0.5 - abs(u)
        k = floor(Int, (2 * a / us + b) * u + c)
        (k < 0 || k > n) && continue

        # squeeze: accept without evaluating the density
        (us >= 0.07 && v <= v_r) && return k

        v = log(v * alpha / (a / (us * us) + b))
        ub = (m + 0.5) * log((m + 1) / (r * (n - m + 1))) +
             (n + 1) * log((n - m + 1) / (n - k + 1)) +
             (k + 0.5) * log(r * (n - k + 1) / (k + 1)) +
             stirling_tail(m) + stirling_tail(n - m) -
             stirling_tail(k) - stirling_tail(n - k)
        v <= ub && return k
    end
end

"""
    binom_rand(rng, n, p)

Generate a `Binomial(n, p)` variate.
"""
function binom_rand(rng, n::Int, p::Float64)
    n <= 0 && return 0
    p <= 0 && return 0
    p >= 1 && return n
    # the samplers assume p <= 1/2; use the symmetry B(n,p) = n - B(n,1-p)
    p > 0.5 && return n - binom_rand(rng, n, 1 - p)
    n * p < 10 && return binom_rand_inversion(rng, n, p)
    return binom_rand_btrs(rng, n, p)
end
