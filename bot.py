import math
from typing import override

from rlbot.flat import AirState, BallAnchor, ControllerState, GamePacket
from rlbot.managers import Bot
from rlbot_flatbuffers import CarAnchor

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import ControlStep, Sequence
from util.vec import Vec3


class MyBot(Bot):
    active_sequence: Sequence | None = None
    boost_pad_tracker: BoostPadTracker = BoostPadTracker()

    @override
    def initialize(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.field_info)

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """
        controls = ControllerState()
        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        if len(packet.balls) == 0:
            # If there are no balls current in the game (likely due to being in a replay), skip this tick.
            return ControllerState()
        # we can now assume there's at least one ball in the match

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            return self.active_sequence.tick(packet)

        # Gather some information about our car and the ball
        my_car = packet.players[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.balls[0].physics.location)

        # By default we will chase the ball, but target_location can be changed later
        target_location = ball_location
        
        target_x = ball_location.x
        target_y = ball_location.y

        dx = target_x - car_location.x
        dy = target_y - car_location.y
        dist_2d = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx) - my_car.physics.rotation.yaw
        angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi
        self.renderer.begin_rendering()
        if self.ball_prediction is not None and len(self.ball_prediction.slices) > 0:
            # Check every 6th slice (every 0.05 seconds) up to 4 seconds ahead
            for i in range(6, min(480, len(self.ball_prediction.slices)), 6):
                ball_slice = self.ball_prediction.slices[i]
                slice_loc = Vec3(ball_slice.physics.location)
                time_until_slice = i / 120.0  # 120 slices per second in v5

                # Only target slices low enough to hit on the ground (or low aerial)
                max_hit_height = 650 if my_car.boost > 45 else 160
                if slice_loc.z > max_hit_height:
                    continue

                # Estimate how long it takes Dawn to drive there
                dist_to_slice = car_location.dist(slice_loc)
                car_speed = max(car_velocity.length(), 1100.0)
                time_to_reach = dist_to_slice / car_speed

                if time_to_reach <= time_until_slice:
                    target_location = slice_loc
                    self.renderer.draw_line_3d(
                        BallAnchor(0), target_location, self.renderer.cyan
                    )
                    break

        
        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(
            CarAnchor(self.index), target_location, self.renderer.white
        )
        self.renderer.draw_string_3d(
            f"Speed: {car_velocity.length():.1f}",
            CarAnchor(self.index),
            1,
            self.renderer.white,
        )
        self.renderer.draw_line_3d(
            target_location - Vec3(0, 0, 50),
            target_location + Vec3(0, 0, 50),
            self.renderer.cyan,
        )
        team_sign = -1 if my_car.team == 0 else 1
        ball_in_our_half = (ball_location.y * team_sign) > -500
        self.renderer.end_rendering()
        
        should_aerial = (ball_in_our_half
                        and 220 < ball_location.z < 850
                        and dist_2d < 2200
                        and my_car.boost > 45
                        and abs(angle_to_target) < 0.35)
        if should_aerial:
            if my_car.air_state == AirState.OnGround:
                controls.jump = True
            else: 
                height_diff = ball_location.z - car_location.z
                target_pitch = math.atan2(height_diff, max(dist_2d, 1.0))
                pitch_error = target_pitch - my_car.physics.rotation.pitch
                
                controls.pitch = max(-1.0, min(1.0, pitch_error * 3.0))
                controls.yaw = max(-1.0, min(1.0, angle_to_target * 2.5))
                controls.boost = True
            return controls
        if 750 < car_velocity.length() < 800:
            # We'll do a front flip if the car is moving at a certain speed.
            return self.begin_front_flip(packet)

        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0
        # You can set more controls if you want, like controls.boost.

        return controls

    def begin_front_flip(self, packet: GamePacket) -> ControllerState:
        # Send some quickchat just for fun
        # There won't be any content of the message for other bots,
        # but "I got it!" will be display for a human to see!
        self.send_match_comm(b"", "I got it!")

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence(
            [
                ControlStep(duration=0.05, controls=ControllerState(jump=True)),
                ControlStep(duration=0.05, controls=ControllerState(jump=False)),
                ControlStep(
                    duration=0.2, controls=ControllerState(jump=True, pitch=-1)
                ),
                ControlStep(duration=0.8, controls=ControllerState()),
            ]
        )

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)


if __name__ == "__main__":
    # Connect to RLBot and run
    # Having the agent id here allows for easier development,
    # as otherwise the RLBOT_AGENT_ID environment variable must be set.
    MyBot("rlbot_community/python_example").run()
