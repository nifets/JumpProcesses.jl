struct SparseIndices
    idx::Vector{Int}
    member::Vector{Bool}
end
SparseIndices(n::Integer) = SparseIndices(Int[], falses(n))

@inline function Base.push!(s::SparseIndices, i)
    @inbounds s.member[i] && return s
    @inbounds s.member[i] = true
    push!(s.idx, i)
    s
end

function Base.empty!(s::SparseIndices)
    @inbounds for i in s.idx
        s.member[i] = false
    end
    empty!(s.idx)
    s
end

Base.iterate(s::SparseIndices, st...) = iterate(s.idx, st...)
Base.isempty(s::SparseIndices) = isempty(s.idx)
Base.length(s::SparseIndices) = length(s.idx)
Base.eltype(::Type{SparseIndices}) = Int

struct DenseIndices
    n::Int
end

@inline Base.push!(s::DenseIndices, i) = s
Base.empty!(s::DenseIndices) = s
Base.iterate(s::DenseIndices, st = 1) = st > s.n ? nothing : (st, st + 1)
Base.isempty(s::DenseIndices) = s.n == 0
Base.length(s::DenseIndices) = s.n
Base.eltype(::Type{DenseIndices}) = Int

##

struct BlendedMassActionJump{M, P} <: AbstractMassActionJump
    inner::M
    policy::P
end

function Base.getproperty(maj::BlendedMassActionJump, f::Symbol)
    (f === :inner || f === :policy) && return getfield(maj, f)
    getproperty(getfield(maj, :inner), f)
end

Base.propertynames(maj::BlendedMassActionJump) =
    (:inner, :policy, propertynames(getfield(maj, :inner))...)

struct ConstantJumpStoich{NS, OS}
    net_stoch::NS
    order::OS
end

jump_stoich(maj, crj_stoich, j) =
    j <= get_num_majumps(maj) ? maj.net_stoch[j] :
    crj_stoich.net_stoch[j - get_num_majumps(maj)]

jump_order(maj, crj_stoich, j) =
    j <= get_num_majumps(maj) ? maj.reactant_stoch[j] :
    (crj_stoich.order === nothing ? () : crj_stoich.order[j - get_num_majumps(maj)])

get_num_majumps(maj::BlendedMassActionJump) = get_num_majumps(maj.inner)

using_params(maj::BlendedMassActionJump) = using_params(maj.inner)

update_parameters!(maj::BlendedMassActionJump, newparams; kwargs...) =
    update_parameters!(maj.inner, newparams; kwargs...)

function self_stoich(maj, crj_stoich, njs)
    map(1:njs) do j
        w = Pair{Int, Float64}[]
        for (spec, o) in jump_order(maj, crj_stoich, j)
            stoch = 0
            for (s, n) in jump_stoich(maj, crj_stoich, j)
                s == spec && (stoch = n)
            end
            iszero(stoch) || push!(w, spec => float(stoch * o))
        end
        w
    end
end

function species_orders(maj, crj_stoich, nspec, njs)
    max_hor = zeros(Float64, nspec)
    max_stoich = ones(Int, nspec)
    nmaj = get_num_majumps(maj)
    for j in 1:njs
        order = jump_order(maj, crj_stoich, j)
        h = 0.0
        for (_, o) in order
            h += float(o)
        end
        for (spec, o) in order
            max_hor[spec] = max(max_hor[spec], h)
            max_stoich[spec] = max(max_stoich[spec], j <= nmaj ? Int(o) : 1)
        end
    end
    max_hor, max_stoich
end

function effective_gi(u, max_hor, max_stoich, i)
    @inbounds h = max_hor[i]
    h <= 1 && return 1.0
    @inbounds k = max_stoich[i]
    k <= 1 && return h
    @inbounds x = float(u[i])
    acc = 0.0
    for r in 0:(k - 1)
        d = x - r
        d <= 0 && return h
        acc += x / d
    end
    (h / k) * acc
end

@inline function evalrxrate(u::AbstractVector{V}, i,
        maj::BlendedMassActionJump) where {V <: Real}
    β = blend(maj.policy, maj.net_stoch[i], maj.reactant_stoch[i], u)
    iszero(β) && return zero(eltype(maj.scaled_rates))
    β * evalrxrate(u, i, maj.inner)
end

@inline function get_majump_brackets(ulow, uhigh, k, maj::BlendedMassActionJump)
    alow, ahigh = get_majump_brackets(ulow, uhigh, k, maj.inner)
    net_stoch, order = maj.net_stoch[k], maj.reactant_stoch[k]
    blend(maj.policy, net_stoch, order, uhigh) * alow,
    blend(maj.policy, net_stoch, order, ulow) * ahigh
end

struct BlendedBracketData{B, T}
    inner::B
    thresholds::Vector{Vector{T}}
end

function get_spec_brackets(bd::BlendedBracketData, i, u::Number)
    lo, hi = get_spec_brackets(bd.inner, i, u)
    @inbounds for θ in bd.thresholds[i]
        if u < θ
            hi = max(u, min(hi, θ - one(θ)))
        else
            lo = min(u, max(lo, θ))
        end
    end
    lo, hi
end

struct AdaptiveTau{T, X}
    epsilon::T
    cap::T
    quantile::T
end
function AdaptiveTau(epsilon, cap; chi::Bool = true, quantile = 0)
    e, c, q = promote(epsilon, cap, quantile)
    0 <= q < 1 || error("AdaptiveTau quantile must lie in [0, 1); got $q.")
    AdaptiveTau{typeof(e), chi}(e, c, q)
end

struct FixedTau{T}
    dt::T
    epsilon::T
end
function FixedTau(dt, epsilon = 0.05)
    isfinite(dt) || error("FixedTau needs a finite step; got dt = $dt.")
    FixedTau(promote(dt, epsilon)...)
end

uses_chi(::AdaptiveTau{T, X}) where {T, X} = X
uses_chi(::FixedTau) = false

struct MinReduce end

struct TopKReduce{T}
    quantile::T
    heap::Vector{T}
end

reducer(tau, ::Type{T}) where {T} = MinReduce()
reducer(tau::AdaptiveTau, ::Type{T}) where {T} =
    iszero(tau.quantile) ? MinReduce() : TopKReduce(T(tau.quantile), T[])

(::MinReduce)(xs) = minimum(xs)

function sift_down!(heap, i, n)
    @inbounds while true
        l = 2i
        l > n && break
        c = l < n && heap[l + 1] > heap[l] ? l + 1 : l
        heap[c] <= heap[i] && break
        heap[i], heap[c] = heap[c], heap[i]
        i = c
    end
end

function (r::TopKReduce)(xs)
    n = length(xs)
    k = clamp(ceil(Int, r.quantile * n), 1, n)
    k == 1 && return minimum(xs)
    heap = r.heap
    length(heap) == k || resize!(heap, k)
    @inbounds for i in 1:k
        heap[i] = xs[i]
    end
    for i in (k ÷ 2):-1:1
        sift_down!(heap, i, k)
    end
    @inbounds for i in (k + 1):n
        xs[i] < heap[1] || continue
        heap[1] = xs[i]
        sift_down!(heap, 1, k)
    end
    @inbounds heap[1]
end

select_tau(sel::FixedTau, p) = sel.dt
select_tau(sel::AdaptiveTau, p) =
    min(sel.cap, max(p.taureduce(p.taus), p.dtmin))

needs_moments(::AdaptiveTau) = true
needs_moments(::FixedTau) = false

abstract type BlendingPolicy end

# `nc` does two jobs: on `net_stoch` it stops a leap exhausting a consumed
# species; on `reactant_stoch` it bounds rate drift while frozen, since one
# event moves `a` by about `order/u[spec]` -- so it tolerates a relative
# change of `1/nc`. Note `nc` must exceed the copy number of any catalyst,
# not just be above 1.
#
# TODO the second job only applies to reactants the *exact* half moves
# mid-window. Restricting it to those (the CRJ-written species, statically)
# would let `nc` be raised for safety without dragging species that only
# leaped reactions touch exact along with it.
# (deriving that set dynamically breaks blend being a function ofu alone).

struct CriticalBlend{T} <: BlendingPolicy
    nc::T
end
CriticalBlend() = CriticalBlend(10)

nc_for(nc::Number, spec) = float(nc)
nc_for(nc::AbstractVector, spec) = @inbounds float(nc[spec])

function blend(policy::CriticalBlend, net_stoch, order, u)
    @inbounds for (spec, change) in net_stoch
        change < 0 && u[spec] < nc_for(policy.nc, spec) * (-change) && return 1.0
    end
    @inbounds for (spec, o) in order
        u[spec] < nc_for(policy.nc, spec) * o && return 1.0
    end
    0.0
end

function blend_thresholds(policy::CriticalBlend, maj, crj_stoich, njs, nspec,
        ::Type{T}) where{T}
    thr = [T[] for _ in 1:nspec]
    for j in 1:njs
        for (spec, change) in jump_stoich(maj, crj_stoich, j)
            change < 0 &&
                push!(thr[spec], ceil(T, nc_for(policy.nc, spec) * (-change)))
        end
        for (spec, o) in jump_order(maj, crj_stoich, j)
            push!(thr[spec], ceil(T, nc_for(policy.nc, spec) * o))
        end
    end
    foreach(v -> unique!(sort!(v)), thr)
    thr
end

function blended_crj(policy, crj_stoich, c, k)
    net_stoch = crj_stoich.net_stoch[k]
    order = crj_stoich.order === nothing ? () : crj_stoich.order[k]
    β(u) = blend(policy, net_stoch, order, u)
    rate = (u, p, t) -> β(u) * c.rate(u, p, t)
    bounds = hasbounds(c) ?
        (ulow, uhigh, u, p, t) -> begin
            b = c.bounds.bounds(ulow, uhigh, u, p, t)
            (; lrate = β(uhigh) * b.lrate, urate = β(ulow) * b.urate)
        end : nothing
    lrate = haslrate(c) ?
        (ulow, uhigh, u, p, t) ->
            (; lrate = β(uhigh) * c.bounds.lrate(ulow, uhigh, u, p, t).lrate) : nothing
    urate = hasurate(c) ?
        (ulow, uhigh, u, p, t) ->
            (; urate = β(ulow) * c.bounds.urate(ulow, uhigh, u, p, t).urate) : nothing
    ConstantRateJump(rate, c.affect!,
        bounds === nothing && lrate === nothing && urate === nothing ? nothing :
        RateBoundFunctions(bounds, lrate, urate))
end

struct AlwaysLeap <: BlendingPolicy end

blended_crj(::AlwaysLeap, crj_stoich, c, k) =
    ConstantRateJump((u, p, t) -> 0.0, c.affect!, nothing)

blend(::AlwaysLeap, net_stoch, order, u) = 0.0

blend_thresholds(::AlwaysLeap, maj, crj_stoich, njs, nspec,
    ::Type{T}) where {T} = [T[] for _ in 1:nspec]

@inline function evalrxrate(u::AbstractVector{V}, i,
    maj::BlendedMassActionJump{<:Any, AlwaysLeap}) where {V <: Real}
    zero(eltype(maj.scaled_rates))
end

@inline function get_majump_brackets(ulow, uhigh, k,
    maj::BlendedMassActionJump{<:Any, AlwaysLeap})
    R = eltype(maj.scaled_rates)
    zero(R), zero(R)
end


struct LinearBlend{T} <: BlendingPolicy
    lower::T
    upper::T
end
LinearBlend() = LinearBlend(10, 100)

function blend_thresholds(policy::LinearBlend, maj, crj_stoich, njs, nspec,
        ::Type{T}) where {T}
    thr = [T[] for _ in 1:nspec]
    lo = ceil(T, policy.lower)
    hi = ceil(T, policy.upper)
    for j in 1:njs
        for stoch in (jump_order(maj, crj_stoich, j), jump_stoich(maj, crj_stoich, j))
            for (spec, _) in stoch
                push!(thr[spec], lo, hi)
            end
        end
    end
    foreach(v -> unique!(sort!(v)), thr)
    thr
end

function involved_min(net_stoch, order, u)
    x = Inf
    @inbounds for (spec, _) in order
        x = min(x, u[spec])
    end
    @inbounds for (spec, _) in net_stoch
        x = min(x, u[spec])
    end
    x
end

function blend(policy::LinearBlend, net_stoch, order, u)
    x = involved_min(net_stoch, order, u)
    x <= policy.lower && return 1.0
    x >= policy.upper && return 0.0
    (policy.upper - x) / (policy.upper - policy.lower)
end


############################################################

mutable struct HybridTauJumpAggregation{T, S, F1, F2, RNG, A, P, U, VJ, C, TS, CS, R} <:
    AbstractSSAJumpAggregator{T, S, F1, F2, RNG}
    # aggregator interface
    next_jump::Int
    prev_jump::Int
    next_jump_time::T
    end_time::T
    ma_jumps::S
    rates::F1
    affects!::F2
    save_positions::Tuple{Bool, Bool}
    rng::RNG
    exact::A
    policy::P
    crj_stoich::C
    # windowing
    tau::TS
    thin_negative::Bool
    dtmin::T
    window_end::T
    exact_due::Bool
    # tau selection
    μ::Vector{Float64}
    σ²::Vector{Float64}
    χ::Vector{Float64}
    self_stoch::Vector{Vector{Pair{Int, Float64}}}
    max_hor::Vector{Float64}
    max_stoich::Vector{Int}
    # rate caches
    rate::Vector{Float64}
    leap_rate::Vector{Float64}
    ulast::U
    stale_rxs::SparseIndices
    vartojumps_map::VJ
    exact_rxs::SparseIndices
    # leap application
    counts::Vector{Int}
    du::U
    changed_specs::CS
    taus::Vector{T}
    taureduce::R
    stale_taus::SparseIndices
end


function HybridTauJumpAggregation(inner::AbstractSSAJumpAggregator{T,S,F1,F2,RNG},
        policy, tau, thin_negative, du, nmaj, vtoj, max_hor, max_stoich, self_stoch,
        crj_stoich, rates) where {T,S,F1,F2,RNG}
    n = length(du)
    njs = nmaj + (crj_stoich === nothing ? 0 : length(crj_stoich.net_stoch))
    changed_specs = policy isa AlwaysLeap && crj_stoich !== nothing ?
                    DenseIndices(n) : SparseIndices(n)
    taureduce = reducer(tau, T)
    HybridTauJumpAggregation{T, S, typeof(rates), F2, RNG, typeof(inner), typeof(policy),
        typeof(du), typeof(vtoj), typeof(crj_stoich), typeof(tau),
        typeof(changed_specs), typeof(taureduce)}(
        inner.next_jump,
        inner.prev_jump,
        inner.next_jump_time,
        inner.end_time,
        inner.ma_jumps,
        rates,
        inner.affects!,
        inner.save_positions,
        inner.rng,
        inner,
        policy,
        crj_stoich,
        tau,
        thin_negative,
        1e-10 * one(T),
        -Inf * one(T),
        false,
        zeros(n),
        zeros(n),
        zeros(n),
        self_stoch,
        max_hor,
        max_stoich,
        zeros(njs),
        zeros(njs),
        zero(du),
        SparseIndices(njs),
        vtoj,
        SparseIndices(njs),
        zeros(Int, njs),
        du,
        changed_specs,
        fill(typemax(T), n),
        taureduce,
        SparseIndices(n))
end

function aggregate(aggregator::HybridTau, u, p, t, end_time, constant_jumps, ma_jumps,
        save_positions, rng; kwargs...)
    net_stoch = get(kwargs, :crj_stoich, nothing)
    crj_stoich = net_stoch === nothing ? nothing :
                 ConstantJumpStoich(net_stoch, get(kwargs, :crj_order, nothing))
    ncrj = net_stoch === nothing ? 0 : length(net_stoch)
    maj = ma_jumps === nothing ? nothing :
          BlendedMassActionJump(ma_jumps, aggregator.policy)
    nrx = get_num_majumps(maj)
    njs = nrx + ncrj
    max_hor, max_stoich = species_orders(maj, crj_stoich, length(u), njs)
    selfs = self_stoich(maj, crj_stoich, njs)
    kw = values(kwargs)
    if njs > 0 && needs_bracketing(aggregator.exact)
        base = get(kw, :bracket_data, nothing)
        base === nothing && (base = BracketData{eltype(u) <: Integer ? Float64 : eltype(u),
                                                eltype(u)}())
        bd = BlendedBracketData(base,
            blend_thresholds(aggregator.policy, maj, crj_stoich, njs,
                length(u), eltype(u)))
        kw = merge(kw, (; bracket_data = bd))
    end

    leap_rates, _ = get_jump_info_fwrappers(u, p, t, constant_jumps)
    exact_jumps = (crj_stoich === nothing || constant_jumps === nothing ||
                   isempty(constant_jumps)) ? constant_jumps :
        [blended_crj(aggregator.policy, crj_stoich, c, k)
         for (k, c) in enumerate(constant_jumps)]
    inner = aggregate(aggregator.exact, u, p, t, end_time, exact_jumps, maj,
        save_positions, rng; kw...)
    vtoj = get(kwargs, :vartojumps_map, nothing)
    if vtoj === nothing && maj !== nothing && isempty(constant_jumps)
        vtoj = var_to_jumps_map(length(u), maj)
    end
    HybridTauJumpAggregation(inner, aggregator.policy, aggregator.tau,
        aggregator.thin_negative, zero(u), nrx, vtoj, max_hor, max_stoich, selfs,
        crj_stoich, leap_rates)
end

# this is only needed because we're implementing the aggregator interface
function sync!(p::HybridTauJumpAggregation)
    inner = p.exact
    p.next_jump = inner.next_jump
    p.prev_jump = inner.prev_jump
    p.end_time = inner.end_time
    nothing
end

update_exact_rates!(exact::DirectJumpAggregation, p, u, params, t) = nothing

update_exact_rates!(exact::RSSAJumpAggregation, p, u, params, t) =
    update_rates!(exact, u, params, t, p.changed_specs.idx)

update_exact_rates!(exact::RSSACRJumpAggregation, p, u, params, t) =
    update_dependent_rates!(exact, u, params, t, p.changed_specs.idx)

function update_exact_rates!(exact, p, u, params, t)
    p.vartojumps_map === nothing && error("HybridTau requires a vartojumps_map to \
        refresh a $(nameof(typeof(exact))) after a leap.")
    rxs = p.exact_rxs
    empty!(rxs)
    @inbounds for spec in p.changed_specs
        for j in p.vartojumps_map[spec]
            push!(rxs, j)
        end
    end
    update_dependent_rates!(exact, u, params, t, sort!(rxs.idx))
end

function concretize_affects!(p::HybridTauJumpAggregation, integrator::SciMLBase.DEIntegrator)
    concretize_affects!(p.exact, integrator)
    p.affects! = p.exact.affects!
    nothing
end

concretize_affects!(p::HybridTauJumpAggregation{T, S, F1, F2},
    ::SciMLBase.DEIntegrator) where {T, S, F1, F2 <: Tuple} = nothing

function initialize!(p::HybridTauJumpAggregation, integrator, u, params, t)
    initialize!(p.exact, integrator, u, params, t)
    sync!(p)
    compute_rates!(p, u, params, t)
    open_window!(p, integrator, u, params, t)
    generate_jumps!(p, integrator, u, params, t)
end

function execute_jumps!(p::HybridTauJumpAggregation, integrator, u, params, t, affects!)
    if p.exact_due
        execute_jumps!(p.exact, integrator, u, params, t, affects!)
    else
        open_window!(p, integrator, u, params, t)
    end
    sync!(p)
    nothing
end

leaps_everything(p) = p.policy isa AlwaysLeap && p.crj_stoich !== nothing

function open_window!(p, integrator, u, params, t)
    refresh_rates!(p, u, params, t)
    needs_moments(p.tau) && update_taus!(p, u, t, p.stale_taus)
    τ = select_tau(p.tau, p)
    draw_leap!(p, τ)
    if p.thin_negative
        while !feasible(u, p.du, p.changed_specs)
            τ /= 2
            if τ <= p.dtmin
                fill!(p.du, zero(eltype(p.du)))
                fill!(p.counts, 0)
                empty!(p.changed_specs)
                DiffEqBase.terminate!(integrator, ReturnCode.Failure)
                break
            end
            thin_leap!(p, 0.5)
        end
    end
    @inbounds for i in p.changed_specs
        u[i] += p.du[i]
    end
    leaps_everything(p) || isempty(p.changed_specs) ||
        update_exact_rates!(p.exact, p, u, params, t)
    p.window_end = t + τ
    nothing
end

function generate_jumps!(p::HybridTauJumpAggregation, integrator, u, params, t)
    if leaps_everything(p)
        p.next_jump_time = p.window_end
        p.exact_due = false
        return nothing
    end
    generate_jumps!(p.exact, integrator, u, params, t)
    sync!(p)
    p.next_jump_time = min(p.exact.next_jump_time, p.window_end)
    p.exact_due = p.exact.next_jump_time <= p.window_end
    nothing
end

function feasible(u, du, changed_specs)
    @inbounds for i in changed_specs
        u[i] + du[i] < 0 && return false
    end
    true
end

advance_to!(integrator, p::HybridTauJumpAggregation, t) =
    advance_to!(integrator, p.exact, t)

jump_rate(p, maj, j, u, params, t) =
    j <= get_num_majumps(maj) ? evalrxrate(u, j, maj.inner) :
    p.rates[j - get_num_majumps(maj)](u, params, t)

jump_leap_rate(policy, net_stoch, order, u, a) =
    (1 - blend(policy, net_stoch, order, u)) * a

njumps(p) = length(p.rate)

function compute_rates!(p, u, params, t)
    (; ma_jumps, crj_stoich, policy, leap_rate, rate, μ, σ², χ, self_stoch,
       ulast, stale_taus, tau) = p
    moments = needs_moments(tau)
    usechi = uses_chi(tau)
    if moments
        fill!(μ, 0.0)
        fill!(σ², 0.0)
        usechi && fill!(χ, 0.0)
    end
    @inbounds for j in 1:njumps(p)
        net_stoch = jump_stoich(ma_jumps, crj_stoich, j)
        a = jump_rate(p, ma_jumps, j, u, params, t)
        rate[j] = a
        leap_rate[j] = jump_leap_rate(policy, net_stoch,
            jump_order(ma_jumps, crj_stoich, j), u, a)
        (moments && !iszero(a)) || continue
        for (spec, ν) in net_stoch
            μ[spec] += ν * a
            σ²[spec] += ν * ν * a
        end
        if usechi
            for (spec, w) in self_stoch[j]
                χ[spec] += w * a
            end
        end
    end
    copyto!(ulast, u)
    if moments
        empty!(stale_taus)
        @inbounds for i in eachindex(u)
            push!(stale_taus, i)
        end
    end
    nothing
end

function refresh_rates!(p, u, params, t)
    p.vartojumps_map === nothing && return compute_rates!(p, u, params, t)
    (; ma_jumps, crj_stoich, policy, leap_rate, rate, μ, σ², χ, self_stoch,
       ulast, stale_rxs, vartojumps_map, stale_taus, max_hor, max_stoich, tau) = p
    njs = njumps(p)
    moments = needs_moments(tau)
    usechi = uses_chi(tau)
    epsilon = tau.epsilon
    empty!(stale_rxs)
    @inbounds for i in eachindex(u)
        u[i] == ulast[i] && continue
        moments && push!(stale_taus, i)
        gi = effective_gi(u, max_hor, max_stoich, i)
        abs(u[i] - ulast[i]) <= 0.5 * epsilon * u[i] / gi && continue
        ulast[i] = u[i]
        for j in vartojumps_map[i]
            j <= njs && push!(stale_rxs, j)
        end
    end
    @inbounds for j in stale_rxs
        net_stoch = jump_stoich(ma_jumps, crj_stoich, j)
        a = jump_rate(p, ma_jumps, j, u, params, t)
        d = a - rate[j]
        rate[j] = a
        leap_rate[j] = jump_leap_rate(policy, net_stoch,
            jump_order(ma_jumps, crj_stoich, j), u, a)
        (moments && !iszero(d)) || continue
        for (spec, ν) in net_stoch
            μ[spec] += ν * d
            σ²[spec] += ν * ν * d
            push!(stale_taus, spec)
        end
        if usechi
            for (spec, w) in self_stoch[j]
                χ[spec] += w * d
                push!(stale_taus, spec)
            end
        end
    end
    nothing
end

function tau_for_species(u, ulast, μ, σ², χ, i, max_hor, max_stoich, t, tau)
    epsilon = tau.epsilon
    τ = typemax(typeof(t))
    @inbounds begin
        gi = effective_gi(u, max_hor, max_stoich, i)
        bound = max(epsilon * u[i] / gi - abs(u[i] - ulast[i]), one(eltype(u)))
        m = abs(μ[i])
        m > 0 && (τ = min(τ, bound / m))
        s = σ²[i]
        s > 0 && (τ = min(τ, bound * bound / s))

        # check if iteration x = x + τμ(x) is stable
        # i.e. τ < 2 / max{λ} where λ are the e-values of the linearisation
        # to avoid computing the full Jacobian, we can use the diagonal entries as an approximation, halving by two to be conservative
        # Jᵢᵢ = Σⱼ νᵢⱼ·∂aⱼ/∂xᵢ = Σⱼ νᵢⱼ·oᵢⱼ·aⱼ/xᵢ = χ[i]/u[i]
        if uses_chi(tau)
            f = abs(χ[i])
            f > 0 && (τ = min(τ, max(u[i], one(eltype(u))) / f))
        end
    end
    τ
end

function update_taus!(p, u, t, idxs)
    (; taus, ulast, μ, σ², χ, max_hor, max_stoich, tau) = p
    @inbounds for i in idxs
        taus[i] = tau_for_species(u, ulast, μ, σ², χ, i, max_hor, max_stoich, t, tau)
    end
    empty!(idxs)
    nothing
end

function draw_leap!(p, τ)
    (; ma_jumps, crj_stoich, leap_rate, counts, du, changed_specs, rng) = p
    @inbounds for i in changed_specs
        du[i] = zero(eltype(du))
    end
    empty!(changed_specs)
    @inbounds for j in 1:njumps(p)
        λ = leap_rate[j] * τ
        counts[j] = λ > 0 ? pois_rand(rng, λ) : 0
        iszero(counts[j]) && continue
        for (spec, stoch) in jump_stoich(ma_jumps, crj_stoich, j)
            du[spec] += stoch * counts[j]
            push!(changed_specs, spec)
        end
    end
end

function thin_leap!(p, r)
    (; ma_jumps, crj_stoich, counts, du, changed_specs, rng) = p
    @inbounds for i in changed_specs
        du[i] = zero(eltype(du))
    end
    empty!(changed_specs)
    @inbounds for j in 1:njumps(p)
        counts[j] = counts[j] > 0 ? binom_rand(rng, counts[j], r) : 0
        iszero(counts[j]) && continue
        for (spec, stoch) in jump_stoich(ma_jumps, crj_stoich, j)
            du[spec] += stoch * counts[j]
            push!(changed_specs, spec)
        end
    end
end
