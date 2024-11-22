# ship_battle.py

import random
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import torch

random.seed(11505050)
np.random.seed(11505050)
torch.manual_seed(11505050)


MAP_SIZE = 256

class Ship:
    def __init__(self, ship_id, ship_type, position, owner):
        self.ship_id = ship_id 
        self.ship_type = ship_type
        self.position = position 
        self.owner = owner 
        self.set_attributes()
        self.alive = True
        self.hit_points_remaining = self.hit_points

    def set_attributes(self):
        if self.ship_type == 'large':
            self.move_range = 4
            self.sonic_missiles = 5
            self.hypersonic_missiles = 4
            self.hit_points = 3  
        elif self.ship_type == 'medium':
            self.move_range = 3
            self.sonic_missiles = 4
            self.hypersonic_missiles = 2
            self.hit_points = 2  
        elif self.ship_type == 'small':
            self.move_range = 2
            self.sonic_missiles = 0
            self.hypersonic_missiles = 3
            self.hit_points = 1 
        else:
            raise ValueError("Unknown ship type")

    def is_sunk(self):
        return self.hit_points_remaining <= 0

    def take_damage(self, damage):
        self.hit_points_remaining -= damage
        if self.is_sunk():
            self.alive = False

    def available_weapons(self):
        weapons = []
        if self.hypersonic_missiles > 0:
            weapons.append('hypersonic')
        if self.sonic_missiles > 0:
            weapons.append('sonic')
        return weapons

class Weapon:
    def __init__(self, weapon_type):
        self.weapon_type = weapon_type
        self.set_attributes()

    def set_attributes(self):
        if self.weapon_type == 'sonic':
            self.attack_pattern = self.get_sonic_attack_pattern()
        elif self.weapon_type == 'hypersonic':
            self.attack_pattern = self.get_hypersonic_attack_pattern()
        else:
            raise ValueError("Unknown weapon type")

    @staticmethod
    def get_sonic_attack_pattern():
        return [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)]

    @staticmethod
    def get_hypersonic_attack_pattern():
        return [(-1, -1), (0, -1), (1, -1),
                (-1,  0), (0,  0), (1,  0),
                (-1,  1), (0,  1), (1,  1)]

class GameMap:
    def __init__(self, size):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]

    def place_ship(self, ship, verbose=False):
        x, y = ship.position
        if self.is_within_bounds(x, y):
            if self.grid[y][x] is None:
                self.grid[y][x] = ship
                if verbose:
                    print(f"Placed Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) at position ({x}, {y})")
            else:
                raise ValueError(f"Position ({x}, {y}) is already occupied by another ship.")
        else:
            raise ValueError(f"Position ({x}, {y}) is out of bounds.")

    def move_ship(self, ship, new_position, verbose=False):
        old_x, old_y = ship.position
        new_x, new_y = new_position
        if self.is_within_bounds(new_x, new_y):
            if self.grid[new_y][new_x] is None:
                self.grid[old_y][old_x] = None
                self.grid[new_y][new_x] = ship
                ship.position = (new_x, new_y)
                if verbose:
                    print(f"Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) moved from ({old_x}, {old_y}) to ({new_x}, {new_y})")
                return True
            else:
                if verbose:
                    print(f"Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) failed to move to ({new_x}, {new_y}) - Position occupied.")
                return False
        else:
            if verbose:
                print(f"Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) failed to move to ({new_x}, {new_y}) - Out of bounds.")
            return False

    def is_within_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def remove_ship(self, ship, verbose=False):
        x, y = ship.position
        self.grid[y][x] = None
        if verbose:
            print(f"Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) removed from position ({x}, {y})")

    def get_ships_by_owner(self, owner):
        ships = []
        for row in self.grid:
            for cell in row:
                if cell is not None and cell.owner == owner and cell.alive:
                    ships.append(cell)
        return ships

    def get_all_ships(self):
        ships = []
        for row in self.grid:
            for cell in row:
                if cell is not None and cell.alive:
                    ships.append(cell)
        return ships

class OriginalStrategy:
    def __init__(self, game_map):
        self.game_map = game_map

    # Attack phase function with greedy focused fire strategy
    def attack_phase(self, attacker_ships, defender_ships, game_map, verbose=False):
        # Filter out sunk ships
        alive_attackers = [ship for ship in attacker_ships if ship.alive]
        alive_defenders = [ship for ship in defender_ships if ship.alive]

        if not alive_attackers or not alive_defenders:
            return [], []  # No ships to attack or defend

        # Sort defenders by priority: small > medium > large
        priority_order = {'small': 1, 'medium': 2, 'large': 3}
        alive_defenders_sorted = sorted(alive_defenders, key=lambda x: priority_order[x.ship_type])

        # Sort attackers by priority: small > medium > large
        attackers_priority_order = {'small': 1, 'medium': 2, 'large': 3}
        alive_attackers_sorted = sorted(alive_attackers, key=lambda x: attackers_priority_order[x.ship_type])

        attacks = []  # Store attack information

        # Keep track of which attackers have already attacked this turn
        attackers_available = alive_attackers_sorted.copy()

        for target_ship in alive_defenders_sorted:
            if not attackers_available:
                break  # No more attackers available

            # Calculate target's reachable positions (current and next turn)
            target_reachable_positions = get_reachable_positions(target_ship, game_map)

            # Define positions that need to be covered
            positions_to_cover = set(target_reachable_positions)

            while positions_to_cover and attackers_available:
                best_attacker = None
                best_weapon = None
                best_attack_position = None
                best_coverage = 0

                # Iterate through available attackers to find the best attack
                for attacker in attackers_available:
                    available_weapons = attacker.available_weapons()
                    if not available_weapons:
                        continue  # This attacker has no available weapons

                    # Prioritize hypersonic over sonic for larger coverage
                    if 'hypersonic' in available_weapons:
                        weapon_type = 'hypersonic'
                    else:
                        weapon_type = 'sonic'

                    weapon = Weapon(weapon_type)

                    # Iterate through target's reachable positions to find attack positions
                    for pos in positions_to_cover:
                        x, y = pos
                        # Potential attack position is the position itself
                        # Alternatively, attacker can attack surrounding positions to cover multiple targets
                        # Here, we choose to attack positions that can cover multiple target positions

                        # Calculate all possible attack positions that can cover this target position
                        for dx, dy in weapon.attack_pattern:
                            attack_x = x + dx
                            attack_y = y + dy
                            if game_map.is_within_bounds(attack_x, attack_y):
                                covered = set()
                                for ddx, ddy in weapon.attack_pattern:
                                    covered_pos = (attack_x + ddx, attack_y + ddy)
                                    if covered_pos in positions_to_cover and game_map.is_within_bounds(*covered_pos):
                                        covered.add(covered_pos)
                                coverage = len(covered)
                                if coverage > best_coverage:
                                    best_coverage = coverage
                                    best_attacker = attacker
                                    best_weapon = weapon
                                    best_attack_position = (attack_x, attack_y)

                if best_attacker and best_weapon and best_attack_position:
                    # Assign the attack
                    attacks.append({
                        'attacker_ship': best_attacker,
                        'weapon': best_weapon,
                        'attack_position': best_attack_position
                    })

                    if verbose:
                        print(f"Ship ID {best_attacker.ship_id} ({best_attacker.owner} {best_attacker.ship_type}) is preparing to use {best_weapon.weapon_type} missile to attack position {best_attack_position} targeting Ship ID {target_ship.ship_id} ({target_ship.owner} {target_ship.ship_type})")

                    # Remove the covered positions
                    for dx, dy in best_weapon.attack_pattern:
                        pos = (best_attack_position[0] + dx, best_attack_position[1] + dy)
                        if pos in positions_to_cover:
                            positions_to_cover.remove(pos)

                    # Remove the attacker from available attackers
                    attackers_available.remove(best_attacker)
                else:
                    # No suitable attacker found to cover remaining positions
                    break

        return attacks, alive_defenders

    # Defense phase function
    def defense_phase(self, defender_ships, game_map, verbose=False):
        alive_defenders = [ship for ship in defender_ships if ship.alive]
        if not alive_defenders:
            if verbose:
                print("No ships available to defend.")
            return  # No ships to defend

        for ship in alive_defenders:
            # Move ship as far as possible while avoiding too close to friendly ships
            move_ship_as_far_as_possible(ship, game_map, verbose=verbose)

def attack_phase(attacker_ships, defender_ships, game_map, original_strategy, verbose=False):
    attacks, _ = original_strategy.attack_phase(attacker_ships, defender_ships, game_map, verbose=verbose)
    return attacks

def defense_phase(defender_ships, game_map, original_strategy, verbose=False):
    original_strategy.defense_phase(defender_ships, game_map, verbose=verbose)

def move_ship_as_far_as_possible(ship, game_map, verbose=False):
    current_x, current_y = ship.position
    possible_moves = []
    for dx in range(-ship.move_range, ship.move_range + 1):
        for dy in range(-ship.move_range, ship.move_range + 1):
            if abs(dx) + abs(dy) == ship.move_range:
                possible_moves.append((dx, dy))

    random.shuffle(possible_moves)

    for move in possible_moves:
        new_x = current_x + move[0]
        new_y = current_y + move[1]

        if not game_map.is_within_bounds(new_x, new_y):
            continue  
        if game_map.grid[new_y][new_x] is not None:
            continue  

        if is_position_too_close(new_x, new_y, ship.owner, game_map):
            if verbose:
                print(f"Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) cannot move to ({new_x}, {new_y}) - Too close to another friendly ship.")
            continue 

        moved = game_map.move_ship(ship, (new_x, new_y), verbose=verbose)
        if moved:
            if verbose:
                print(f"Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) moved to ({new_x}, {new_y}) as far as possible.")
            return 

    if verbose:
        print(f"Ship ID {ship.ship_id} ({ship.owner} {ship.ship_type}) could not move and stays at ({current_x}, {current_y}).")

def is_position_too_close(x, y, owner, game_map):
    MIN_MANDIST = 3

    friendly_ships = game_map.get_ships_by_owner(owner)
    for ship in friendly_ships:
        if ship.position == (x, y):
            continue  
        distance = manhattan_distance((x, y), ship.position)
        if distance < MIN_MANDIST:
            return True  
    return False  

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def resolve_attacks(attacks, game_map, verbose=False):
    weapon_areas = []  
    for attack in attacks:
        attacker_ship = attack['attacker_ship']
        weapon = attack['weapon']
        attack_position = attack['attack_position']
        affected_positions = perform_attack(attacker_ship, weapon, attack_position, game_map, verbose=verbose)
        weapon_areas.extend(affected_positions)
    return weapon_areas

def perform_attack(attacker_ship, weapon, attack_position, game_map, verbose=False):
    attack_positions = []
    tx, ty = attack_position
    for dx, dy in weapon.attack_pattern:
        x, y = tx + dx, ty + dy
        if game_map.is_within_bounds(x, y):
            attack_positions.append((x, y))
    hit = False
    for x, y in attack_positions:
        target = game_map.grid[y][x]
        if target and target.alive:
            target.take_damage(1)
            hit = True
            relation = 'enemy' if target.owner != attacker_ship.owner else 'friendly'
            if verbose:
                print(f"Ship ID {attacker_ship.ship_id} attacked position {attack_position}, hit {relation} ship ID {target.ship_id} ({target.owner} {target.ship_type})!")
                print(f"Target ship ID {target.ship_id} remaining HP: {target.hit_points_remaining}")
            if target.is_sunk():
                game_map.remove_ship(target, verbose=verbose)
                if verbose:
                    print(f"Target ship ID {target.ship_id} has been destroyed!")
    if not hit and verbose:
        print(f"Ship ID {attacker_ship.ship_id} attacked position {attack_position}, but hit nothing.")
    if weapon.weapon_type == 'sonic':
        attacker_ship.sonic_missiles -= 1
        if verbose:
            print(f"Ship ID {attacker_ship.ship_id} remaining sonic missiles: {attacker_ship.sonic_missiles}")
    elif weapon.weapon_type == 'hypersonic':
        attacker_ship.hypersonic_missiles -= 1
        if verbose:
            print(f"Ship ID {attacker_ship.ship_id} remaining hypersonic missiles: {attacker_ship.hypersonic_missiles}")
    return attack_positions  

def get_reachable_positions(ship, game_map):
    reachable_positions = set()
    x, y = ship.position
    for dx in range(-ship.move_range, ship.move_range + 1):
        for dy in range(-ship.move_range, ship.move_range + 1):
            if abs(dx) + abs(dy) > ship.move_range:
                continue  
            new_x = x + dx
            new_y = y + dy
            if game_map.is_within_bounds(new_x, new_y):
                reachable_positions.add((new_x, new_y))
    return reachable_positions

def visualize_game(game_map, round_number, weapon_areas=None):
    data = np.zeros((game_map.size, game_map.size))
    for ship in game_map.get_all_ships():
        x, y = ship.position
        if ship.owner == 'A':
            if ship.ship_type == 'large':
                data[y][x] = 1  
            elif ship.ship_type == 'medium':
                data[y][x] = 2
            elif ship.ship_type == 'small':
                data[y][x] = 3
        elif ship.owner == 'B':
            if ship.ship_type == 'large':
                data[y][x] = 4
            elif ship.ship_type == 'medium':
                data[y][x] = 5
            elif ship.ship_type == 'small':
                data[y][x] = 6

    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = ListedColormap(['black', 'blue', 'cyan', 'lightblue', 'red', 'orange', 'pink'])
    ax.imshow(data, cmap=cmap, interpolation='none')

    if weapon_areas:
        weapon_layer = np.zeros((game_map.size, game_map.size))
        for x, y in weapon_areas:
            weapon_layer[y][x] = 1  
        ax.imshow(weapon_layer, cmap=ListedColormap(['none', 'red']), alpha=0.3, interpolation='none')

    ax.set_title(f"Ship positions after round {round_number}")
    ax.axis('off')
    plt.show()

# 定義 Gym 環境
class ShipBattleEnvGym(gym.Env):
    """
    自定義 Gym 環境。代理人控制 Player A，Player B 使用 OriginalStrategy。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ShipBattleEnvGym, self).__init__()
        # Actions: 0=Move Up, 1=Move Down, 2=Move Left, 3=Move Right, 4=Stay, 5=Use Sonic Missile, 6=Use Hypersonic Missile
        self.action_space = spaces.Discrete(7)

        # 定義觀察空間：地圖為 MAP_SIZE x MAP_SIZE 的網格，具有 6 個通道
        # 通道：
        # 0: Player A Large
        # 1: Player A Medium
        # 2: Player A Small
        # 3: Player B Large
        # 4: Player B Medium
        # 5: Player B Small
        self.observation_space = spaces.Box(low=0, high=1, shape=(MAP_SIZE, MAP_SIZE, 6), dtype=np.float32)

        self.reset()

    def reset(self):
        self.game_map, self.ships = initialize_game(verbose=False)
        self.done = False
        self.reward = 0
        self.info = {}
        self.step_count = 0
        self.max_steps = 100  # 每個回合的最大步數

        # OriginalStrategy
        self.original_strategy = OriginalStrategy(self.game_map)

        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((MAP_SIZE, MAP_SIZE, 6), dtype=np.float32)
        for ship in self.game_map.get_all_ships():
            x, y = ship.position
            if ship.owner == 'A':
                if ship.ship_type == 'large':
                    obs[y][x][0] = 1  # Player A Large
                elif ship.ship_type == 'medium':
                    obs[y][x][1] = 1  # Player A Medium
                elif ship.ship_type == 'small':
                    obs[y][x][2] = 1  # Player A Small
            elif ship.owner == 'B':
                if ship.ship_type == 'large':
                    obs[y][x][3] = 1  # Player B Large
                elif ship.ship_type == 'medium':
                    obs[y][x][4] = 1  # Player B Medium
                elif ship.ship_type == 'small':
                    obs[y][x][5] = 1  # Player B Small
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), self.reward, self.done, self.info

        self.step_count += 1

        player_A_ships = self.game_map.get_ships_by_owner('A')
        if not player_A_ships:
            self.done = True
            self.reward = -10  
            self.info = {'result': 'Player A has no ships left.'}
            return self._get_obs(), self.reward, self.done, self.info

        agent_ship = player_A_ships[0]  

        if action < 5:
            move_mapping = {
                0: (0, -1),  
                1: (0, 1),   
                2: (-1, 0),  
                3: (1, 0),   
                4: (0, 0)    
            }
            dx, dy = move_mapping[action]
            new_x = agent_ship.position[0] + dx
            new_y = agent_ship.position[1] + dy
            if self.game_map.is_within_bounds(new_x, new_y) and self.game_map.grid[new_y][new_x] is None:
                self.game_map.move_ship(agent_ship, (new_x, new_y), verbose=False)
        elif action == 5:
            # Sonic Missile
            if agent_ship.sonic_missiles > 0:
                target_ship = self.select_target('B')
                if target_ship:
                    weapon = Weapon('sonic')
                    perform_attack(agent_ship, weapon, target_ship.position, self.game_map, verbose=False)
                    agent_ship.sonic_missiles -= 1
        elif action == 6:
            # Hypersonic Missile
            if agent_ship.hypersonic_missiles > 0:
                target_ship = self.select_target('B')
                if target_ship:
                    weapon = Weapon('hypersonic')
                    perform_attack(agent_ship, weapon, target_ship.position, self.game_map, verbose=False)
                    agent_ship.hypersonic_missiles -= 1

        # Player B （OriginalStrategy）
        player_B_ships = self.game_map.get_ships_by_owner('B')
        player_A_ships = self.game_map.get_ships_by_owner('A')
        attacks = attack_phase(player_B_ships, player_A_ships, self.game_map, self.original_strategy, verbose=False)
        defense_phase(player_A_ships, self.game_map, self.original_strategy, verbose=False)
        weapon_areas_B = resolve_attacks(attacks, self.game_map, verbose=False)

        ships_A_remaining = len([ship for ship in self.game_map.get_ships_by_owner('A') if ship.alive])
        ships_B_remaining = len([ship for ship in self.game_map.get_ships_by_owner('B') if ship.alive])
        self.reward = ships_A_remaining - ships_B_remaining

        if ships_B_remaining == 0:
            self.done = True
            self.reward += 10  
            self.info = {'result': 'Player A wins!'}
        elif ships_A_remaining == 0:
            self.done = True
            self.reward -= 10  
            self.info = {'result': 'Player A loses!'}
        elif self.step_count >= self.max_steps:
            self.done = True
            self.reward -= 5  
            self.info = {'result': 'Maximum steps reached.'}
        else:
            self.done = False
            self.info = {}

        return self._get_obs(), self.reward, self.done, self.info

    def select_target(self, target_owner):
        target_ships = self.game_map.get_ships_by_owner(target_owner)
        if not target_ships:
            return None
        agent_ship = self.game_map.get_ships_by_owner('A')[0]
        min_distance = float('inf')
        closest_ship = None
        for ship in target_ships:
            distance = manhattan_distance(agent_ship.position, ship.position)
            if distance < min_distance:
                min_distance = distance
                closest_ship = ship
        return closest_ship

    def render(self, mode='human'):

        visualize_game(self.game_map, round_number=self.step_count)

    def close(self):
        pass

def initialize_game(verbose=False):
    game_map = GameMap(MAP_SIZE)
    ships = []
    ship_id = 1   

    # Player A ship configuration: 4 large, 2 medium, 2 small ships
    player_A_ships = [
        ('large', 4),
        ('medium', 2),
        ('small', 2),
    ]

    # Player B ship configuration: 3 large, 3 medium, 3 small ships
    player_B_ships = [
        ('large', 3),
        ('medium', 3),
        ('small', 3),
    ]

    # Place Player A ships on the left half
    for ship_type, count in player_A_ships:
        for _ in range(count):
            position = get_random_position(game_map, side='left')
            ship = Ship(ship_id, ship_type, position, owner='A')
            game_map.place_ship(ship, verbose=verbose)
            ships.append(ship)
            ship_id += 1

    # Place Player B ships on the right half
    for ship_type, count in player_B_ships:
        for _ in range(count):
            position = get_random_position(game_map, side='right')
            ship = Ship(ship_id, ship_type, position, owner='B')
            game_map.place_ship(ship, verbose=verbose)
            ships.append(ship)
            ship_id += 1

    return game_map, ships


def get_random_position(game_map, side=None):
    max_attempts = 10000
    attempts = 0
    while attempts < max_attempts:
        if side == 'left':
            x = random.randint(0, game_map.size // 2 - 1)
        elif side == 'right':
            x = random.randint(game_map.size // 2, game_map.size - 1)
        else:
            x = random.randint(0, game_map.size - 1)
        y = random.randint(0, game_map.size - 1)
        if game_map.grid[y][x] is None:
            return (x, y)
        attempts += 1
    raise RuntimeError("Failed to find an empty position on the map.")


def visualize_game(game_map, round_number, weapon_areas=None):
    data = np.zeros((game_map.size, game_map.size))
    for ship in game_map.get_all_ships():
        x, y = ship.position
        if ship.owner == 'A':
            if ship.ship_type == 'large':
                data[y][x] = 1  
            elif ship.ship_type == 'medium':
                data[y][x] = 2
            elif ship.ship_type == 'small':
                data[y][x] = 3
        elif ship.owner == 'B':
            if ship.ship_type == 'large':
                data[y][x] = 4
            elif ship.ship_type == 'medium':
                data[y][x] = 5
            elif ship.ship_type == 'small':
                data[y][x] = 6

    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = ListedColormap(['black', 'blue', 'cyan', 'lightblue', 'red', 'orange', 'pink'])
    ax.imshow(data, cmap=cmap, interpolation='none')

    if weapon_areas:
        weapon_layer = np.zeros((game_map.size, game_map.size))
        for x, y in weapon_areas:
            weapon_layer[y][x] = 1  
        ax.imshow(weapon_layer, cmap=ListedColormap(['none', 'red']), alpha=0.3, interpolation='none')

    ax.set_title(f"Ship positions after round {round_number}")
    ax.axis('off')
    plt.show()

def run_multiple_simulations(model, env, num_simulations=100):
    A = 0
    B = 0 
    for i in range(1, num_simulations + 1):
        if i == 1:
            print(f"\n### Simulation {i} ###")
            remaining_A, remaining_B = run_single_game(verbose=True, visualize=True)
        else:
            remaining_A, remaining_B = run_single_game(verbose=False, visualize=False)

        A += remaining_A
        B += remaining_B

    print(f"\n### Aggregated Results After {num_simulations} Simulations ###")
    print(f"Average remaining ships of Player A per simulation: {A}")
    print(f"Average remaining ships of Player A per simulation: {B}")


def train_a2c_agent():

    env = ShipBattleEnvGym()
    env = DummyVecEnv([lambda: env])
    model = A2C(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="./a2c_ship_battle_tensorboard/",
        device='cpu' 
    )

    print("Starting training...")
    model.learn(total_timesteps=10000)  
    print("Training completed.")

    model.save("a2c_ship_battle")
    print("Model saved as 'a2c_ship_battle'.")
    env.close()

def evaluate_a2c_agent(model_path, num_episodes=100):

    env = ShipBattleEnvGym()
    model = A2C.load(model_path)
    env = DummyVecEnv([lambda: env])
    run_multiple_simulations(model, env, num_simulations=num_episodes)
    env.close()

def run_single_game(verbose=False, visualize=False):
    game_map, ships = initialize_game(verbose=verbose)
    total_rounds = 10
    original_strategy = OriginalStrategy(game_map)

    for round_number in range(1, total_rounds + 1):
        if verbose:
            print(f"\n===== Round {round_number} begins =====\n")

        if verbose:
            print("---- Player A's Attack ----")
        attacker_ships = game_map.get_ships_by_owner('A')
        defender_ships = game_map.get_ships_by_owner('B')
        attacks = attack_phase(attacker_ships, defender_ships, game_map, original_strategy, verbose=verbose)
        if verbose:
            print("\n---- Player B's Defense ----")
        defense_phase(defender_ships, game_map, original_strategy, verbose=verbose)
        if verbose:
            print("\n---- Resolving Player A's Attacks ----")
        weapon_areas_A = resolve_attacks(attacks, game_map, verbose=verbose)

        defender_ships = game_map.get_ships_by_owner('B')
        if not any(ship.alive for ship in defender_ships):
            if verbose:
                print("All Player B's ships have been destroyed. Player A wins!")
                visualize_game(game_map, round_number, weapon_areas_A)
            return len([ship for ship in game_map.get_ships_by_owner('A') if ship.alive]), len([ship for ship in game_map.get_ships_by_owner('B') if ship.alive])

        if verbose:
            print("\n---- Player B's Attack ----")
        attacker_ships = game_map.get_ships_by_owner('B')
        defender_ships = game_map.get_ships_by_owner('A')
        attacks = attack_phase(attacker_ships, defender_ships, game_map, original_strategy, verbose=verbose)
        if verbose:
            print("\n---- Player A's Defense ----")
        defense_phase(defender_ships, game_map, original_strategy, verbose=verbose)
        if verbose:
            print("\n---- Resolving Player B's Attacks ----")
        weapon_areas_B = resolve_attacks(attacks, game_map, verbose=verbose)

        defender_ships = game_map.get_ships_by_owner('A')
        if not any(ship.alive for ship in defender_ships):
            if verbose:
                print("All Player A's ships have been destroyed. Player B wins!")
                visualize_game(game_map, round_number, weapon_areas_B)
            return len([ship for ship in game_map.get_ships_by_owner('A') if ship.alive]), len([ship for ship in game_map.get_ships_by_owner('B') if ship.alive])

        if visualize:
            combined_weapon_areas = weapon_areas_A + weapon_areas_B
            visualize_game(game_map, round_number, combined_weapon_areas)

    else:
        ships_A_remaining = len([ship for ship in game_map.get_ships_by_owner('A') if ship.alive])
        ships_B_remaining = len([ship for ship in game_map.get_ships_by_owner('B') if ship.alive])
        if verbose:
            print(f"Game over!")
            print(f"Player A remaining ships: {ships_A_remaining}")
            print(f"Player B remaining ships: {ships_B_remaining}")
        if visualize:
            visualize_game(game_map, round_number)
        return ships_A_remaining, ships_B_remaining
    


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate A2C agent for Ship Battle.")
    parser.add_argument('--train', action='store_true', help="Train the A2C agent.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the trained A2C agent.")
    parser.add_argument('--model_path', type=str, default="a2c_ship_battle", help="Path to the trained model.")
    parser.add_argument('--num_episodes', type=int, default=100, help="Number of evaluation episodes.")

    args = parser.parse_args()

    if args.train:
        train_a2c_agent()

    if args.evaluate:
        evaluate_a2c_agent(args.model_path, args.num_episodes)

if __name__ == "__main__":
    main()
