"""
Per-track velocity/acceleration in real units (m/s, m/s^2), from a smoothed ground-plane
trajectory - not raw frame-to-frame differences, which are too noisy (detection jitter) to
differentiate directly. See LEVEL2_PLAN.md.

Depends on pipeline/geo_reference.py for the ground-plane (x_m, y_m) coordinates this
operates on - this module only smooths + differentiates, it doesn't project pixels itself.
"""
import numpy as np
from scipy.signal import savgol_filter


def compute_kinematics(times_s, x_m, y_m, speed_window: int = 9, accel_window: int = 21,
                        polyorder: int = 2):
    """
    times_s, x_m, y_m: equal-length sequences for one track, in time order (real-world
    ground-plane metres, e.g. from GroundPlaneHomography.pixel_to_ground on each frame's
    ground_contact_point).

    Acceleration (a 2nd derivative) needs a noticeably wider smoothing window than velocity
    to be trustworthy - confirmed empirically via this module's own self-test: with equal
    windows, tiny synthetic jitter (0.02m std) gave a fine speed estimate (~0.07 m/s error)
    but a badly noisy acceleration estimate (~1.16 m/s^2 error against a true 2.0 m/s^2, a
    real numerical property of 2nd-derivative estimation, not a bug). Using a wider window
    specifically for acceleration fixes this, at the cost of responsiveness to sudden
    acceleration changes - acceptable for typical traffic kinematics, not for hard braking
    events on the scale of a few frames.

    Returns (speed_mps, accel_mps2) arrays the same length as the input, or (None, None) if
    the track is too short to smooth reliably (fewer points than accel_window) - callers
    should skip kinematics for short tracks rather than fall back to noisy raw differencing.
    """
    times_s = np.asarray(times_s, dtype=float)
    x_m = np.asarray(x_m, dtype=float)
    y_m = np.asarray(y_m, dtype=float)
    n = len(times_s)
    if n < accel_window:
        return None, None

    def odd_capped(window):
        window = window if window % 2 == 1 else window + 1
        return min(window, n if n % 2 == 1 else n - 1)

    speed_window = odd_capped(speed_window)
    accel_window = odd_capped(accel_window)
    dt = float(np.median(np.diff(times_s)))

    vx = savgol_filter(x_m, speed_window, polyorder, deriv=1, delta=dt)
    vy = savgol_filter(y_m, speed_window, polyorder, deriv=1, delta=dt)
    speed_mps = np.sqrt(vx**2 + vy**2)

    ax = savgol_filter(x_m, accel_window, polyorder, deriv=2, delta=dt)
    ay = savgol_filter(y_m, accel_window, polyorder, deriv=2, delta=dt)
    accel_mps2 = np.sqrt(ax**2 + ay**2)

    return speed_mps, accel_mps2


if __name__ == "__main__":
    # Self-test against a synthetic straight-line, constant-acceleration trajectory -
    # doesn't need real calibration data, just checks the smoothing/differentiation math.
    dt = 1 / 30
    n = 60
    t = np.arange(n) * dt
    true_accel = 2.0  # m/s^2
    true_v0 = 5.0      # m/s
    x = true_v0 * t + 0.5 * true_accel * t**2
    y = np.zeros(n)
    # Add small jitter, like real detection noise would produce.
    rng = np.random.default_rng(0)
    x_noisy = x + rng.normal(0, 0.02, n)

    speed, accel = compute_kinematics(t, x_noisy, y)
    assert speed is not None
    mid = slice(15, 45)  # avoid edge effects for the sanity check
    speed_err = np.abs(speed[mid] - (true_v0 + true_accel * t[mid])).mean()
    accel_err = np.abs(accel[mid] - true_accel).mean()
    print(f"Self-test: mean speed error {speed_err:.3f} m/s, mean accel error {accel_err:.3f} m/s^2")
    assert speed_err < 0.5, "speed estimate too far off for a clean synthetic trajectory"
    assert accel_err < 1.0, "accel estimate too far off for a clean synthetic trajectory"
    print("Self-test OK.")
