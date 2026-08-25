"""
DeskSonar Physical Laptop Geometry, 20cm Geofence & 3D Bounding Dimensions Calculator
Calculates:
1. Real-time Hand 3D Dimensions: Length (Depth), Width (Breadth), Height (Thickness) in cm
2. Strict 20cm Origin Geofence Spherical Radius: R = sqrt(X^2 + Y^2 + Z^2) <= 20cm
3. Laptop Screen Tilt Angle (theta_tilt) & Mic Height (H_mic)
4. Win32 Desktop Screen Pixel Mapping within the 20cm interaction volume
"""
import math
import dataclasses
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


@dataclasses.dataclass
class HandBoundingBox3D:
    length_cm: float              # Depth along Z-axis (cm)
    width_cm: float               # Lateral span along X-axis (cm)
    height_cm: float              # Vertical thickness along Y-axis (cm)
    origin_distance_cm: float     # Spherical distance R from mic origin (cm)
    is_in_20cm_geofence: bool     # True if R <= 20.0 cm
    centroid_3d_m: Tuple[float, float, float] # (X, Y, Z) meters from mic origin


@dataclasses.dataclass
class LaptopGeometryProfile:
    screen_tilt_deg: float          # e.g. 108.0 degrees
    mic_height_m: float             # e.g. 0.205 meters above desk
    desk_plane_distance_m: float    # Specular desk reflection distance
    active_tracking_fov_x_m: float  # Horizontal interaction width (0.24m)
    active_tracking_fov_z_m: float  # Depth interaction reach (0.04m to 0.20m)
    calibrated_at: float


class SpatialPlaneCalibrator:
    """
    Computes real-time laptop geometry, strict 10-20cm interaction geofence, and 3D hand bounding box.
    Features dual-boundary Schmitt trigger hysteresis:
    - Entry Zone: 10.0 cm <= R <= 19.0 cm (0.10m to 0.19m)
    - Retention Zone: 8.5 cm <= R <= 21.5 cm (0.085m to 0.215m)
    """

    def __init__(
        self,
        default_tilt_deg: float = 108.0,
        screen_length_m: float = 0.22,
        speed_of_sound: float = 343.4,
        geofence_radius_m: float = 0.20  # Strict 20cm
    ):
        self.screen_length = screen_length_m
        self.c = speed_of_sound
        self.geofence_radius_m = geofence_radius_m
        self._in_interaction_zone: bool = False

        self.profile = LaptopGeometryProfile(
            screen_tilt_deg=default_tilt_deg,
            mic_height_m=self.screen_length * math.sin(math.radians(default_tilt_deg)),
            desk_plane_distance_m=0.12,
            active_tracking_fov_x_m=0.24,   # +/- 12cm within 20cm bubble
            active_tracking_fov_z_m=0.20,   # 4cm to 20cm forward
            calibrated_at=0.0
        )

    def is_within_interaction_zone(self, range_m: Optional[float]) -> bool:
        """
        Strict 10-20 cm Radial Geofencing with Schmitt Trigger Hysteresis:
        - Entry Zone: 0.10 m <= R <= 0.19 m (10 to 19 cm)
        - Retention Zone: 0.085 m <= R <= 0.215 m (8.5 to 21.5 cm)
        If range_m is None or negative, returns False and resets state.
        """
        if range_m is None or range_m < 0.0:
            self._in_interaction_zone = False
            return False

        if self._in_interaction_zone:
            # Retention Zone with hysteresis
            self._in_interaction_zone = (0.085 <= range_m <= 0.215)
        else:
            # Entry Zone
            self._in_interaction_zone = (0.100 <= range_m <= 0.190)

        return self._in_interaction_zone

    def reset_zone_state(self) -> None:
        self._in_interaction_zone = False

    def auto_calibrate_from_impulse_response(self, cir_profile: np.ndarray, range_axis: np.ndarray) -> LaptopGeometryProfile:
        """
        Extracts the static specular desk reflection peak to calculate real-time screen tilt.
        """
        valid_idx = np.where((range_axis >= 0.08) & (range_axis <= 0.28))[0]
        if len(valid_idx) > 0:
            sub_profile = cir_profile[valid_idx]
            peak_local_idx = int(np.argmax(sub_profile))
            desk_dist = float(range_axis[valid_idx[peak_local_idx]])

            ratio = max(0.1, min(0.95, desk_dist / (2.0 * self.screen_length)))
            tilt_from_vert = math.degrees(math.asin(ratio))
            screen_tilt = 90.0 + tilt_from_vert
            mic_h = self.screen_length * math.sin(math.radians(screen_tilt))

            self.profile = LaptopGeometryProfile(
                screen_tilt_deg=round(screen_tilt, 1),
                mic_height_m=round(mic_h, 3),
                desk_plane_distance_m=round(desk_dist, 3),
                active_tracking_fov_x_m=0.24,
                active_tracking_fov_z_m=0.20,
                calibrated_at=0.0
            )

        return self.profile

    def calculate_3d_bounding_box(
        self,
        range_m: Optional[float],
        azimuth_deg: float,
        phase_disp_mm: float,
        range_profile_db: np.ndarray,
        cfar_curve_db: np.ndarray,
        range_axis_m: np.ndarray,
        rdm_slice: Optional[np.ndarray] = None
    ) -> HandBoundingBox3D:
        """
        Calculates real-time Length (Depth), Width (Breadth), and Height (Thickness)
        of the physical hand reflecting ultrasound within the 20cm geofence.
        """
        above_cfar = np.where(range_profile_db > cfar_curve_db)[0]

        if range_m is None:
            if len(above_cfar) > 0 and len(range_axis_m) > 0:
                peak_idx = above_cfar[int(np.argmax(range_profile_db[above_cfar]))]
                range_m = float(range_axis_m[min(peak_idx, len(range_axis_m) - 1)])
            else:
                return HandBoundingBox3D(
                    length_cm=0.0,
                    width_cm=0.0,
                    height_cm=0.0,
                    origin_distance_cm=999.0,
                    is_in_20cm_geofence=False,
                    centroid_3d_m=(0.0, round(self.profile.mic_height_m, 3), 0.0)
                )

        az_rad = math.radians(azimuth_deg)
        tilt_rad = math.radians(self.profile.screen_tilt_deg - 90.0)

        # 1. 3D Centroid Coordinates from Origin (0,0,0)
        x_m = range_m * math.sin(az_rad)
        y_m = self.profile.mic_height_m + (phase_disp_mm * 0.001)
        z_m = range_m * math.cos(az_rad) * math.cos(tilt_rad)

        origin_dist_m = math.sqrt(x_m**2 + y_m**2 + z_m**2)
        origin_dist_cm = round(origin_dist_m * 100.0, 1)
        is_in_geofence = origin_dist_m <= self.geofence_radius_m

        # 2. Length (Depth Span Delta Z in cm)
        # Find continuous range bins exceeding CFAR threshold around target peak
        if len(above_cfar) > 1:
            dr = range_axis_m[1] - range_axis_m[0] if len(range_axis_m) > 1 else 0.01
            span_bins = (above_cfar[-1] - above_cfar[0] + 1)
            raw_len_cm = span_bins * dr * 100.0
            length_cm = max(4.0, min(18.0, raw_len_cm))
        else:
            length_cm = 8.5  # Typical human palm depth

        # 3. Width (Lateral Span Delta X in cm)
        # Stereo beam spread W = 2 * R * tan(delta_theta / 2)
        spread_angle_deg = 18.0  # Palm lateral angular aperture
        width_cm = max(4.0, min(16.0, 2.0 * range_m * 100.0 * math.tan(math.radians(spread_angle_deg / 2.0))))

        # 4. Height (Vertical Elevation Thickness Delta Y in cm)
        height_cm = max(2.5, min(8.0, 3.5 + abs(phase_disp_mm) * 0.2))

        return HandBoundingBox3D(
            length_cm=round(float(length_cm), 1),
            width_cm=round(float(width_cm), 1),
            height_cm=round(float(height_cm), 1),
            origin_distance_cm=float(origin_dist_cm),
            is_in_20cm_geofence=bool(is_in_geofence),
            centroid_3d_m=(round(float(x_m), 3), round(float(y_m), 3), round(float(z_m), 3))
        )

    def project_3d_to_screen(
        self,
        range_m: Optional[float],
        azimuth_deg: float,
        phase_disp_mm: float,
        screen_width_px: int,
        screen_height_px: int
    ) -> Tuple[int, int]:
        """
        Transforms 3D hand position inside the 20cm geofence into Windows screen pixel coordinates.
        """
        if range_m is None:
            return screen_width_px // 2, screen_height_px // 2

        az_rad = math.radians(azimuth_deg)
        tilt_rad = math.radians(self.profile.screen_tilt_deg - 90.0)

        # X: Lateral across desk [-0.10m, +0.10m] within 20cm bubble
        x_desk = range_m * math.sin(az_rad)
        half_fov_x = 0.10  # 10cm half-width
        norm_x = (x_desk + half_fov_x) / (2.0 * half_fov_x)
        norm_x = max(0.0, min(1.0, norm_x))
        px_x = int(norm_x * screen_width_px)

        # Z: Forward reach [0.04m, 0.20m]
        z_forward = range_m * math.cos(az_rad) * math.cos(tilt_rad)
        z_min = 0.04
        z_max = 0.20
        norm_y = 1.0 - ((z_forward - z_min) / (z_max - z_min))
        norm_y -= (phase_disp_mm * 0.015)
        norm_y = max(0.0, min(1.0, norm_y))
        px_y = int(norm_y * screen_height_px)

        return px_x, px_y
