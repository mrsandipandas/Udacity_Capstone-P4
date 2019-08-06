from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband,
                 decel_limit, accel_limit, wheel_radius, wheel_base,
                 steer_ratio, max_lat_accel, max_steer_angle, sampling_freq):
        self.dt = 1.0/sampling_freq
        self.brake_deadband = brake_deadband
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.fuel_capacity = fuel_capacity
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.min_steer_speed = 2.5


        self.yaw_controller      = YawController(wheel_base, steer_ratio, self.min_steer_speed, max_lat_accel, max_steer_angle)
        self.accel_controller    = PID(kp=0.45, ki=0.02, kd=0.01, mn=decel_limit, mx=accel_limit)
        self.throttle_controller = PID(kp=1.0, ki=0.001, kd=0.10, mn=0.0, mx=0.2)
        self.lowpass_filter      = LowPassFilter(0.15, self.dt)

    # Time 7.08 udacity
    def control(self, curr_vel, dbw_enabled, linear_vel, angular_vel):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        vel_error = linear_vel - curr_vel
        raw_accel = self.accel_controller.step(vel_error, self.dt)

        self.lowpass_filter.filt(raw_accel)
        accel = self.lowpass_filter.get()

        brake = 0.0
        throttle = 0.0
        steer = self.yaw_controller.get_steering(linear_vel, angular_vel, curr_vel)

        if accel > 0.0:
            throttle = self.throttle_controller.step(accel, self.dt)

        if linear_vel == 0 and curr_vel < 0.1:
            throttle = 0
            brake = 400

        elif throttle < .1 and accel < 0:
            throttle = 0
            accel = max(accel,self.decel_limit)
            brake = abs(accel) * (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius * 2

        return throttle, brake, steer
