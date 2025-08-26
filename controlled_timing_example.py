from werkzeug.sansio.multipart import LINE_BREAK

from graphics.agent import Agent

# Simulation parameters

# Define the agent's behavior in units per second
LINEAR_SPEED_MPS = 0.01     # 1 cm/s
ANG_SPEED_MPS = 45.0        # 45 degrees/s

# Define the simulation's internal clock
# Controlled step frequency, for instance 100 Hz.
# 1000 Hz would make this script execute 10 times faster, but the simulation would be identical.
SIM_FREQ_HZ = 100
FIXED_DT = 1.0 / SIM_FREQ_HZ

# Define how long the experiment should run in *simulated time*
TOTAL_SIMULATED_TIME_SEC = 5.0

# Setup simulation

agent = Agent()
simulation_time = 0.0

print("Starting headless simulation...")
print(f" - Fixed Delta Time: {FIXED_DT}s ({SIM_FREQ_HZ} Hz)")
print(f" - Total Duration:   {TOTAL_SIMULATED_TIME_SEC}s")
print("-" * 40)
print(f"Time: {simulation_time:.2f}s | Pos: {agent.position} | Yaw: {agent.yaw:.1f}°")


# Run simulation loop (as fast as possible)

while simulation_time < TOTAL_SIMULATED_TIME_SEC:

    agent.dt(FIXED_DT).translate(agent.forward * LINEAR_SPEED_MPS).rotate_axis(ANG_SPEED_MPS, 'yaw')

    # Advance simulation clock
    simulation_time += FIXED_DT

    # Log the state at intervals
    if int(simulation_time * 10) % 10 == 0: # print roughly every second (for a 100 Hz sim)
         print(f"Time: {simulation_time:.2f}s | Pos: {agent.position} | Yaw: {agent.yaw:.1f}°")

print("-" * 40)
print("Simulation complete.")
print(f"Final Time: {simulation_time:.2f}s")
print(f"Final Position: {agent.position}")
print(f"Final Yaw: {agent.yaw:.2f}°")

# Verification:
# The position will be the result of moving along an arc
# Expected simulated time: 5 seconds
# Expected final yaw: 5.0s * 45 deg/s = 225 degrees (or -135 degrees, same thing)