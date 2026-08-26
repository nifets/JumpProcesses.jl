abstract type AbstractPropensityBounds end

@inline eval_rate(rx, state, rates, params, t) = rates[rx](state, params, t)

function jump_lower_bound end
function jump_upper_bound end

@inline function jump_bounds(pb, rx, ulow, uhigh, rates, params, t)
    (
        jump_lower_bound(pb, rx, ulow, uhigh, rates, params, t),
        jump_upper_bound(pb, rx, ulow, uhigh, rates, params, t)
    )
end

struct IncreasingBounds <: AbstractPropensityBounds end
jump_lower_bound(::IncreasingBounds, rx, ulow, uhigh, rates, params, t) =
    eval_rate(rx, ulow, rates, params, t)
jump_upper_bound(::IncreasingBounds, rx, ulow, uhigh, rates, params, t) =
    eval_rate(rx, uhigh, rates, params, t)


struct MonotoneBounds <: AbstractPropensityBounds end
jump_lower_bound(::MonotoneBounds, rx, ulow, uhigh, rates, params, t) =
    min(eval_rate(rx, ulow, rates, params, t), eval_rate(rx, uhigh, rates, params, t))
jump_upper_bound(::MonotoneBounds, rx, ulow, uhigh, rates, params, t) =
    max(eval_rate(rx, ulow, rates, params, t), eval_rate(rx, uhigh, rates, params, t))

@inline function jump_bounds(::MonotoneBounds, rx, ulow, uhigh, rates, params, t)
    a = eval_rate(rx, ulow, rates, params, t)
    b = eval_rate(rx, uhigh, rates, params, t)
    minmax(a, b)
end


struct DirectionalBounds{V} <: AbstractPropensityBounds
    dirs::Vector{Vector{Pair{Int, Int8}}}
    corner_scratch::V
end

function DirectionalBounds{T}(dirs, num_species::Integer) where {T}
    DirectionalBounds(dirs, zeros(T, num_species))
end

function corner!(pb::DirectionalBounds, rx, ulow, uhigh, upper::Bool)
    v = pb.corner_scratch
    @inbounds for (spec, d) in pb.dirs[rx]
        high = (d > 0) == upper
        v[spec] = high ? uhigh[spec] : ulow[spec]
    end
    v
end

jump_lower_bound(pb::DirectionalBounds, rx, ulow, uhigh, rates, params, t) =
    eval_rate(rx, corner!(pb, rx, ulow, uhigh, false), rates, params, t)
jump_upper_bound(pb::DirectionalBounds, rx, ulow, uhigh, rates, params, t) =
    eval_rate(rx, corner!(pb, rx, ulow, uhigh, true), rates, params, t)

struct ExplicitBounds{L, U} <: AbstractPropensityBounds
    lrate::L
    urate::U
end

jump_lower_bound(pb::ExplicitBounds, rx, ulow, uhigh, rates, params, t) =
    pb.lrate[rx](ulow, uhigh, params, t)

jump_upper_bound(pb::ExplicitBounds, rx, ulow, uhigh, rates, params, t) =
    pb.urate[rx](ulow, uhigh, params, t)
