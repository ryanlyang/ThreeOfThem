import torch
from engine import PhysicsEngine
from model.GNN import PhysicsGNN
from model.actor import CelestialActor
from model.critic import CentralizedCritic
from torch.distributions import Normal

GAMMA=.2
GAE_LAMBDA=.3

engine = PhysicsEngine()
gnn = PhysicsGNN()
actors = [CelestialActor() for _ in range(3)]
critic = CentralizedCritic()

optimizer = torch.optim.Adam()

def train_active_orchestration(engine, gnn, actors, critic, optimizer,
    epochs=15, k_epochs=15, batch_size=64, old_data_size=1000):
    for epoch in range(epochs):
        graph_0 = engine.reset()
        done = False

        old_data = {
            "nodes": [],
            "edges": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "is_done": [],
        }
        
        for i in range(old_data_size):
            with torch.no_grad():
                h = gnn(graph_0["nodes"], graph_0["edges"])
                
                actions = torch.zeros((3,3))
                log_prob = 0
                for i, actor in enumerate(actors):
                    mean, std = actor(h[i])
                    distrib = Normal(mean, std)

                    a_i = distrib.sample()
                    actions[i] = a_i
                    log_prob += distrib.log_prob(a_i).sum(dim=-1)
                value = critic(h).unsqueeze(-1)
            graph_1, reward, done, info = engine.step(actions)
            
            old_data["nodes"].append(graph_0["nodes"])
            old_data["edges"].append(graph_0["edges"])
            old_data["actions"].append(actions)
            old_data["log_probs"].append(log_prob)
            old_data["rewards"].append(reward)
            old_data["values"].append(value)
            old_data["is_done"].append(0 if done else 1)

            graph_0 = graph_1         

            # If we detect a collision we exit the simulation
            if done: break
        
        # Number of generated time_steps
        B = len(old_data["nodes"])
        
        # Compute advantages        
        advantages= torch.zeros((B,))
        returns = []
        last_gae_lam = 0

        for i in range(B):
            if i == 0:
                with torch.no_grad():
                    value_1 = critic(gnn(graph_0["nodes"], graph_0["edges"]))
            else:
                value_1 = old_data["values"][B - i]
            delta = old_data['rewards'][B - 1 - i] + GAMMA * value_1 * old_data["is_done"] - old_data['values'][B - 1 - i]
            advantages = last_gae_lam = delta + GAMMA * GAE_LAMBDA * old_data["is_done"] * last_gae_lam

            returns.insert(0, advantages + old_data["values"][B - 1 - i])
        returns = torch.tensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Iterate policy changes over rollout data
        old_data["nodes"] = torch.tensor(old_data["nodes"])
        old_data["edges"] = torch.tensor(old_data["edges"])
        old_data["actions"] = torch.tensor(old_data["actions"])
        old_data["log_probs"] = torch.tensor(old_data["log_probs"])
        old_data["rewards"] = torch.tensor(old_data["rewards"])
        old_data["values"] = torch.tensor(old_data["values"])

        for e in range(k_epochs):
            idx = torch.randperm(B)
            # Mini batches on the old data
            for b_i in range(0, B, batch_size):
                b_nodes = old_data["nodes"][b_i: b_i + batch_size]
                b_edges = old_data["edges"][b_i: b_i + batch_size]
                b_actions = old_data["actions"][b_i: b_i + batch_size]

                cur_h = gnn(b_nodes, b_edges)

                cur_log_prob = 0
                for i, actor in enumerate(actors):
                    mean, std = actor(cur_h[:,i])
                    dist = Normal(mean, std)

                    cur_log_prob += dist.log_prob(b_actions[:,i]).sum(dim=-1)

        print(f"Epoch {epoch} complete. Reward: {reward}")

if __name__ == "__main__":
    train_active_orchestration(engine, gnn, actors, critic, epochs=100)