import os
import json
from opengnc.simulation import MissionSimulator, SimulationLogger, ScenarioConfig

# Example Dynamics (1D Double Integrator)
def propagator(t, state, dt, control):
    # state = [position, velocity]
    if control is None:
        u = 0.0
    else:
        u = control
    
    # Simple Euler integration
    pos = state[0] + state[1] * dt
    vel = state[1] + u * dt
    return [pos, vel]

def sensor_model(t, state):
    # Perfect sensor for simplicity
    return state[0]

def estimator(t, measurements):
    # Estimate is just the position
    return [measurements, 0.0]

def controller(t, est):
    # Proportional controller to reach position 10.0
    des_pos = 10.0
    kp = 2.0
    return kp * (des_pos - est[0])

def main():
    # 1. Provide an example scenario config
    scenario_data = {
        "simulation": {
            "t0": 0.0,
            "tf": 10.0,
            "dt": 0.1
        },
        "initial_state": [0.0, 0.0]
    }
    
    config_path = "examples/sim_config.json"
    with open(config_path, "w") as f:
        json.dump(scenario_data, f, indent=4)
        
    config = ScenarioConfig(config_path)
    t0 = config.get("simulation.t0", 0.0)
    tf = config.get("simulation.tf", 10.0)
    dt = config.get("simulation.dt", 0.1)
    x0 = config.get("initial_state", [0.0, 0.0])

    # 2. Setup Logger
    log_path = "examples/sim_log"
    logger = SimulationLogger(log_path)

    # 3. Setup Simulator
    sim = MissionSimulator(
        propagator=propagator,
        sensor_model=sensor_model,
        estimator=estimator,
        controller=controller,
        logger=logger
    )

    # Schedule an event to change target mid-way
    def change_target():
        print(f"[{sim.time:.2f}] Event triggered: Changing target to 20.0!")
        global controller
        def new_controller(t, est):
            des_pos = 20.0
            kp = 2.0
            return kp * (des_pos - est[0])
        sim.controller = new_controller

    sim.schedule_event(5.0, change_target)

    # 4. Run simulation
    print("Starting simulation...")
    sim.initialize(t0, x0)
    sim.run(tf, dt)
    print("Simulation finished.")

    # 5. Save logs
    logger.save_csv()
    print(f"Logs saved to {log_path}.csv")

    # Read back and print the final state
    final_state = logger.history[-1]
    print(f"Final state at t={final_state['time']:.2f}: {final_state['state']}")

if __name__ == "__main__":
    main()




