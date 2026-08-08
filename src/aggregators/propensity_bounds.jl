abstract type AbstractPropensityBounds end

@inline eval_rate(rx, state, rates, params, t) = rates[rx](state, params, t)

function jump_lower_bound end
function jump_upper_bound end

struct IncreasingBounds <: AbstractPropensityBounds end
jump_lower_bound(::IncreasingBounds, rx, ulow, uhigh, rs, p, t) =
    eval_rate(rx, ulow, rs, p, t)
jump_upper_bound(::IncreasingBounds, rx, ulow, uhigh, rs, p, t) =
    eval_rate(rx, uhigh, rs, p, t)


struct MonotoneBounds <: AbstractPropensityBounds end
jump_lower_bound(::MonotoneBounds, rx, ulow, uhigh, rs, p, t) =
    min(eval_rate(rx, ulow, rs, p, t), eval_rate(rx, uhigh, rs, p, t))
jump_upper_bound(::MonotoneBounds, rx, ulow, uhigh, rs, p, t) =
    max(eval_rate(rx, ulow, rs, p, t), eval_rate(rx, uhigh, rs, p, t))


struct DirectionalBounds{V} <: AbstractPropensityBounds
    dirs::Vector{Vector{Pair{Int, Int8}}}
    corner::V
end

function corner!(pb::DirectionalBounds, rx, ulow, uhigh, upper::Bool)
    v = pb.corner
    @inbounds for (spec, d) in pb.dirs[rx]
        high = (d > 0) == upper
        v[spec] = high ? uhigh[spec] : ulow[spec]
    end
    v
end

jump_lower_bound(pb::DirectionalBounds, rx, ulow, uhigh, rs, p, t) =
    eval_rate(rx, corner!(pb, rx, ulow, uhigh, false), rs, p, t)
jump_upper_bound(pb::DirectionalBounds, rx, ulow, uhigh, rs, p, t) =
    eval_rate(rx, corner!(pb, rx, ulow, uhigh, true), rs, p, t)

struct ExplicitBounds{L, U} <: AbstractPropensityBounds
    lrate::L
    urate::U
end

jump_lower_bound(pb::ExplicitBounds, rx, ulow, uhigh, rates, p, t) =
    pb.lrate[rx](ulow, uhigh, p, t)

jump_upper_bound(pb::ExplicitBounds, rx, ulow, uhigh, rates, p, t) =
    pb.urate[rx](ulow, uhigh, p, t)
