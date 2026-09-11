#ifndef ERF_FIRE_TYPES_H
#define ERF_FIRE_TYPES_H

#include <AMReX_REAL.H>

#include <cmath>

namespace ERFFire
{

/**
 * Two-dimensional vector in physical horizontal coordinates.
 */
struct FireVec2 {
    amrex::Real x{};
    amrex::Real y{};
};

inline FireVec2
operator+ (const FireVec2& a, const FireVec2& b) noexcept
{
    return {a.x + b.x, a.y + b.y};
}

inline FireVec2
operator- (const FireVec2& a, const FireVec2& b) noexcept
{
    return {a.x - b.x, a.y - b.y};
}

inline FireVec2
operator* (const FireVec2& v, amrex::Real scalar) noexcept
{
    return {v.x * scalar, v.y * scalar};
}

inline FireVec2
operator* (amrex::Real scalar, const FireVec2& v) noexcept
{
    return v * scalar;
}

inline FireVec2
operator/ (const FireVec2& v, amrex::Real scalar) noexcept
{
    return {v.x / scalar, v.y / scalar};
}

inline amrex::Real
dot (const FireVec2& a, const FireVec2& b) noexcept
{
    return a.x * b.x + a.y * b.y;
}

inline amrex::Real
norm_squared (const FireVec2& v) noexcept
{
    return dot(v, v);
}

inline amrex::Real
norm (const FireVec2& v) noexcept
{
    return std::sqrt(norm_squared(v));
}

} // namespace ERFFire

#endif // ERF_FIRE_TYPES_H
