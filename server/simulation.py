import matplotlib.pyplot as plt
import os
import uuid
import time
from common.game_object import Environment

def run_simulation(agent, settings, lives=3, sid=None, socketio=None):

    env = Environment(settings)
    state_dim = len(env.reset())
    if not hasattr(agent, 'actor'):
        from server.app import DDPGAgent
        agent = DDPGAgent(state_dim, 2, settings)
    results = []
    energy_history = []

    for life in range(lives):
        state = env.reset()
        episode_energy = []
        total_energy = env.agent.energy
        food_eaten = 0
        done = False
        step = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action, track_approach=False)
            state = next_state
            total_energy = env.agent.energy
            food_eaten = env.agent.food_eaten
            episode_energy.append(total_energy)
            step += 1

            if sid and socketio:
                render_data = env.get_render_data()
                render_data['life'] = life + 1
                render_data['step'] = step
                socketio.emit('simulation_frame', render_data, room=sid)
                time.sleep(0.016)

        results.append({
            'life': life + 1,
            'steps_survived': step,
            'final_energy': total_energy,
            'food_eaten': food_eaten
        })
        energy_history.append(episode_energy)

    plt.figure(figsize=(10, 6))
    for i, energies in enumerate(energy_history):
        plt.plot(energies, label=f'Life {i+1}')
    plt.title('Energy Accumulation Over Time')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join('server', 'simulations', f'simulation_{uuid.uuid4()}.png')
    plt.savefig(plot_path)
    plt.close()

    return results, plot_path