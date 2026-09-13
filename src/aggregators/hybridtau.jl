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

##

struct BlendedMassActionJump{M, P} <: AbstractMassActionJump
    inner::M
    policy::P
end

function Base.getproperty(m::BlendedMassActionJump, f::Symbol)
    (f === :inner || f === :policy) && return getfield(m, f)
    getproperty(getfield(m, :inner), f)
end

Base.propertynames(m::BlendedMassActionJump) =
    (:inner, :policy, propertynames(getfield(m, :inner))...)

get_num_majumps(m::BlendedMassActionJump) = get_num_majumps(m.inner)

using_params(m::BlendedMassActionJump) = using_params(m.inner)

update_parameters!(m::BlendedMassActionJump, newparams; kwargs...) =
    update_parameters!(m.inner, newparams; kwargs...)

function firings_left(m, i, u)
    L = typemax(Int)
    @inbounds for (spec, change) in m.net_stoch[i]
        change < 0 && (L = min(L, floor(Int, u[spec] / -change)))
    end
    L
end

@inline function evalrxrate(u::AbstractVector{V}, i, m::BlendedMassActionJump) where {V <: Real}
    β = blend(m.policy, m, i, u)
    iszero(β) && return zero(eltype(m.scaled_rates))
    β * evalrxrate(u, i, m.inner)
end

@inline function get_majump_brackets(ulow, uhigh, k, m::BlendedMassActionJump)
    alow, ahigh = get_majump_brackets(ulow, uhigh, k, m.inner)
    blend(m.policy, m, k, uhigh) * alow, blend(m.policy, m, k, ulow) * ahigh
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

function blend(policy::CriticalBlend, m, i, u)
    @inbounds for (spec, change) in m.net_stoch[i]
        change < 0 && u[spec] < float(policy.nc) * (-change) && return 1.0
    end
    @inbounds for (spec, order) in m.reactant_stoch[i]
        u[spec] < float(policy.nc) * order && return 1.0
    end
    0.0
end

function blend_thresholds(policy::CriticalBlend, maj, nspec, ::Type{T}) where{T}
    thr = [T[] for _ in 1:nspec]
    for j in 1:get_num_majumps(maj)
        for (spec, change) in maj.net_stoch[j]
            change < 0 && push!(thr[spec], ceil(T, float(policy.nc) * (-change)))
        end
        for (spec, order) in maj.reactant_stoch[j]
            push!(thr[spec], ceil(T, float(policy.nc) * order))
        end
    end
    foreach(v -> unique!(sort!(v)), thr)
    thr
end

struct AlwaysLeap <: BlendingPolicy end

blend(::AlwaysLeap, m, i, u) = 0.0

blend_thresholds(::AlwaysLeap, maj, nspec, ::Type{T}) where {T} = [T[] for _ in 1:nspec]

@inline function evalrxrate(u::AbstractVector{V}, i,
    m::BlendedMassActionJump{<:Any, AlwaysLeap}) where {V <: Real}
    zero(eltype(m.scaled_rates))
end

@inline function get_majump_brackets(ulow, uhigh, k,
    m::BlendedMassActionJump{<:Any, AlwaysLeap})
    R = eltype(m.scaled_rates)
    zero(R), zero(R)
end


struct LinearBlend{T} <: BlendingPolicy
    lower::T
    upper::T
end
LinearBlend() = LinearBlend(10, 100)

function blend_thresholds(policy::LinearBlend, maj, nspec, ::Type{T}) where {T}
    thr = [T[] for _ in 1:nspec]
    lo = ceil(T, policy.lower)
    hi = ceil(T, policy.upper)
    for j in 1:get_num_majumps(maj)
        for stoch in (maj.reactant_stoch[j], maj.net_stoch[j])
            for (spec, _) in stoch
                push!(thr[spec], lo, hi)
            end
        end
    end
    foreach(v -> unique!(sort!(v)), thr)
    thr
end

function involved_min(m, i, u)
    x = Inf
    @inbounds for (spec, _) in m.reactant_stoch[i]
        x = min(x, u[spec])
    end
    @inbounds for (spec, _) in m.net_stoch[i]
        x = min(x, u[spec])
    end
    x
end

function blend(policy::LinearBlend, m, i, u)
    x = involved_min(m, i, u)
    x <= policy.lower && return 1.0
    x >= policy.upper && return 0.0
    (policy.upper - x) / (policy.upper - policy.lower)
end


############################################################

mutable struct HybridTauJumpAggregation{T, S, F1, F2, RNG, A, P, U, VJ, CS} <:
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
    crj_stoich::CS
    # windowing
    dt::T
    dtmin::T
    epsilon::T
    window_end::T
    exact_due::Bool
    # tau selection
    μ::Vector{Float64}
    σ²::Vector{Float64}
    max_hor::Vector{Int}
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
    changed_specs::SparseIndices
end


function HybridTauJumpAggregation(inner::AbstractSSAJumpAggregator{T,S,F1,F2,RNG},
        policy, dt, epsilon, du, counts, vtoj, max_hor, max_stoich,
        crj_stoich) where {T,S,F1,F2,RNG}
    n = length(du)
    nmaj = length(counts)
    njs = nmaj + (crj_stoich === nothing ? 0 : length(crj_stoich))
    HybridTauJumpAggregation{T, S, F1, F2, RNG, typeof(inner), typeof(policy), typeof(du),
        typeof(vtoj), typeof(crj_stoich)}(
        inner.next_jump,
        inner.prev_jump,
        inner.next_jump_time,
        inner.end_time,
        inner.ma_jumps,
        inner.rates,
        inner.affects!,
        inner.save_positions,
        inner.rng,
        inner,
        policy,
        crj_stoich,
        convert(T, dt),
        1e-10 * one(T),
        convert(T, epsilon),
        -Inf * one(T),
        false,
        zeros(n),
        zeros(n),
        max_hor,
        max_stoich,
        zeros(njs),
        zeros(nmaj),
        zero(du),
        SparseIndices(njs),
        vtoj,
        SparseIndices(njs),
        counts,
        du,
        SparseIndices(n))
end

function aggregate(aggregator::HybridTau, u, p, t, end_time, constant_jumps, ma_jumps,
        save_positions, rng; kwargs...)
    maj = ma_jumps === nothing ? nothing :
          BlendedMassActionJump(ma_jumps, aggregator.policy)
    nrx = maj === nothing ? 0 : get_num_majumps(maj)
    if maj === nothing
        max_hor = Int[]; max_stoich = Int[]
    else
        hor = compute_hor(maj.reactant_stoch, nrx)
        max_hor, max_stoich = precompute_reaction_conditions(maj.reactant_stoch, hor, length(u), nrx)
    end
    kw = values(kwargs)
    if maj !== nothing && needs_bracketing(aggregator.exact)
        base = get(kw, :bracket_data, nothing)
        base === nothing && (base = BracketData{eltype(u) <: Integer ? Float64 : eltype(u),
                                                eltype(u)}())
        bd = BlendedBracketData(base,
            blend_thresholds(aggregator.policy, maj, length(u), eltype(u)))
        kw = merge(kw, (; bracket_data = bd))
    end

    inner = aggregate(aggregator.exact, u, p, t, end_time, constant_jumps, maj,
        save_positions, rng; kw...)
    vtoj = get(kwargs, :vartojumps_map, nothing)
    if vtoj === nothing && maj !== nothing && isempty(constant_jumps)
        vtoj = var_to_jumps_map(length(u), maj)
    end
    HybridTauJumpAggregation(inner, aggregator.policy, aggregator.dt, aggregator.epsilon,
        zero(u), zeros(Int, nrx), vtoj, max_hor, max_stoich,
        get(kwargs, :crj_stoich, nothing))
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
    recompute_rates!(p, p.ma_jumps, u, params, t)
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

function open_window!(p, integrator, u, params, t)
    refresh_rates!(p, p.ma_jumps, u, params, t)
    τ = min(p.dt, tau_from_moments(u, p.μ, p.σ², eachindex(u), p.max_hor,
        p.max_stoich, t, p.epsilon, p.dtmin))
    draw_leap!(p, p.ma_jumps, τ)
    while !feasible(u, p.du, p.changed_specs)
        τ /= 2
        if τ <= p.dtmin
            fill!(p.du, zero(eltype(p.du)))
            fill!(p.counts, 0)
            empty!(p.changed_specs)
            DiffEqBase.terminate!(integrator, ReturnCode.Failure)
            break
        end
        thin_leap!(p, p.ma_jumps, 0.5)
    end
    @inbounds for i in p.changed_specs
        u[i] += p.du[i]
    end
    isempty(p.changed_specs) || update_exact_rates!(p.exact, p, u, params, t)
    p.window_end = t + τ
    nothing
end

function generate_jumps!(p::HybridTauJumpAggregation, integrator, u, params, t)
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

refresh_rates!(p, ::Nothing, u, params, t) = nothing

recompute_rates!(p, ::Nothing, u, params, t) = nothing

jump_rate(p, m, j, u, params, t) = j <= get_num_majumps(m) ? evalrxrate(u, j, m.inner) :
                                   p.rates[j - get_num_majumps(m)](u, params, t)

jump_stoich(p, m, j) = j <= get_num_majumps(m) ? m.net_stoch[j] :
                       p.crj_stoich[j - get_num_majumps(m)]

jump_leap_rate(p, m, j, u, a) = j <= get_num_majumps(m) ?
                                (1 - blend(m.policy, m, j, u)) * a : 0.0

njumps(p) = length(p.rate)

function recompute_rates!(p, m::BlendedMassActionJump, u, params, t)
    (; leap_rate, rate, μ, σ², ulast) = p
    fill!(μ, 0.0)
    fill!(σ², 0.0)
    @inbounds for j in 1:njumps(p)
        a = jump_rate(p, m, j, u, params, t)
        rate[j] = a
        j <= get_num_majumps(m) && (leap_rate[j] = jump_leap_rate(p, m, j, u, a))
        iszero(a) && continue
        for (spec, ν) in jump_stoich(p, m, j)
            μ[spec] += ν * a
            σ²[spec] += ν * ν * a
        end
    end
    copyto!(ulast, u)
    nothing
end

function refresh_rates!(p, m::BlendedMassActionJump, u, params, t)
    p.vartojumps_map === nothing && return recompute_rates!(p, m, u, params, t)
    (; leap_rate, rate, μ, σ², ulast, stale_rxs, vartojumps_map) = p
    njs = njumps(p)
    empty!(stale_rxs)
    @inbounds for i in eachindex(u)
        u[i] == ulast[i] && continue
        ulast[i] = u[i]
        for j in vartojumps_map[i]
            j <= njs && push!(stale_rxs, j)
        end
    end
    @inbounds for j in stale_rxs
        a = jump_rate(p, m, j, u, params, t)
        d = a - rate[j]
        rate[j] = a
        j <= get_num_majumps(m) && (leap_rate[j] = jump_leap_rate(p, m, j, u, a))
        iszero(d) && continue
        for (spec, ν) in jump_stoich(p, m, j)
            μ[spec] += ν * d
            σ²[spec] += ν * ν * d
        end
    end
    nothing
end

function tau_from_moments(u, μ, σ², idxs, max_hor, max_stoich, t, epsilon, dtmin)
    τ = typemax(typeof(t))
    @inbounds for i in idxs
        gi = compute_gi(u, max_hor, max_stoich, i, t)
        bound = max(epsilon * u[i] / gi, one(eltype(u)))
        m = abs(μ[i])
        m > 0 && (τ = min(τ, bound / m))
        s = σ²[i]
        if s > 0
            τ = min(τ, bound * bound / s)
            τ = min(τ, max(u[i], one(eltype(u))) / s)
        end
    end
    max(τ, dtmin)
end

draw_leap!(p, ::Nothing, τ) = nothing

function draw_leap!(p, m::BlendedMassActionJump, τ)
    (; leap_rate, counts, du, changed_specs, rng) = p
    @inbounds for i in changed_specs
        du[i] = zero(eltype(du))
    end
    empty!(changed_specs)
    @inbounds for j in 1:get_num_majumps(m)
        λ = leap_rate[j] * τ
        counts[j] = λ > 0 ? pois_rand(rng, λ) : 0
        iszero(counts[j]) && continue
        for (spec, stoch) in m.net_stoch[j]
            du[spec] += stoch * counts[j]
            push!(changed_specs, spec)
        end
    end
end

thin_leap!(p, ::Nothing, r) = nothing

function thin_leap!(p, m::BlendedMassActionJump, r)
    (; counts, du, changed_specs, rng) = p
    @inbounds for i in changed_specs
        du[i] = zero(eltype(du))
    end
    empty!(changed_specs)
    @inbounds for j in 1:get_num_majumps(m)
        counts[j] = counts[j] > 0 ? binom_rand(rng, counts[j], r) : 0
        iszero(counts[j]) && continue
        for (spec, stoch) in m.net_stoch[j]
            du[spec] += stoch * counts[j]
            push!(changed_specs, spec)
        end
    end
end
