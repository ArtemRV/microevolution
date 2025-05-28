# tests/test_physics_processor.py
import unittest
from unittest.mock import Mock # unittest.mock.Mock is generally preferred over manual mocks for flexibility
import numpy as np

from common.physics_processor import PhysicsProcessor
from common.game_object import GameObject, Organism, Food, Obstacle # Import Food and Obstacle too

# A simple mock for the Environment
class MockEnvironment:
    def __init__(self, dish_radius, dish_center):
        self.dish_radius = dish_radius
        self.dish_center = np.array(dish_center, dtype=float)
        # Potentially add other attributes if PhysicsProcessor methods expect them from env
        # For example, if any method tried to access self.env.settings directly
        # self.settings = {} # Or a more detailed mock settings if needed by env itself

class TestPhysicsProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_settings = {
            'general': { # Added a general section for dish_radius if it comes from settings in real env
                'dish_radius': 100.0,
                'width': 200.0, # Example value
                'height': 200.0 # Example value
            },
            'organism': {
                'radius': 5.0,
                'max_acceleration': 1.0,
                'max_speed': 2.0,
                'initial_energy': 100.0, # Added for Organism
                'visible_obstacle': 5,   # Added for Organism
                'visible_food': 5,       # Added for Organism
                'energy_per_food': 20.0, # Added for Organism
                'energy_per_step': 0.1,  # Added for Organism
                'max_energy': 200.0,     # Added for Organism
                'random_start': False,   # Added for Organism
            },
            'food': {
                'radius': 3.0,
                'enabled': True, # Assuming these might be checked
                'quantity': 5
            },
            'obstacle': {
                'radius': 7.0,
                'max_speed': 0.0, # Obstacles are static in some contexts
                'enabled': True,
                'quantity': 3
            },
            'rewards': { # Added for Organism init
                'energy': {'value': -0.1}
            }
            # other general settings...
        }
        # Using dish_center that allows objects to be placed without immediate collision for simple tests
        self.mock_env = MockEnvironment(
            dish_radius=self.mock_settings['general']['dish_radius'], 
            dish_center=[100.0, 100.0] # Centered within a 200x200 area
        )
        # If PhysicsProcessor expects env.settings:
        # self.mock_env.settings = self.mock_settings

        self.physics_processor = PhysicsProcessor(self.mock_env, self.mock_settings)

        # Example of creating a simple generic game object for tests
        # This can be used as a template or base for more specific mocks in actual tests
        self.test_game_object = GameObject.__new__(GameObject)
        self.test_game_object.env = self.mock_env # GameObject __init__ assigns this
        self.test_game_object.settings = self.mock_settings # GameObject __init__ assigns this
        self.test_game_object.object_type = 'organism' # Example
        self.test_game_object.radius = self.mock_settings['organism']['radius']
        self.test_game_object.pos = np.array([100.0, 100.0]) # Center of the mock env
        self.test_game_object.vel = np.array([1.0, 0.0])
        self.test_game_object.render_component = None

        # Example Organism (might be more useful to create fresh in each test)
        # self.test_organism = Organism(self.mock_env, self.mock_settings) 
        # Note: Organism __init__ calls _initialize_position which might need more env setup
        # or mocking of is_position_free if used directly.
        # For unit testing PhysicsProcessor methods, we often just need objects with pos, vel, radius.

    def _create_test_game_object(self, pos, vel, radius=5.0):
        obj = GameObject.__new__(GameObject)
        obj.pos = np.array(pos, dtype=float)
        obj.vel = np.array(vel, dtype=float)
        obj.radius = float(radius)
        obj.env = self.mock_env # process_generic_movement uses self.env.dish_center and self.env.dish_radius
        obj.settings = self.mock_settings # Not strictly used by process_generic_movement but good for consistency
        obj.object_type = 'test_object'
        obj.render_component = None
        return obj

    def test_process_generic_movement_no_collision(self):
        # Object starts at center, moves towards top-right, no collision expected
        initial_pos = [100.0, 100.0]
        initial_vel = [1.0, 1.0]
        game_object = self._create_test_game_object(pos=initial_pos, vel=initial_vel)

        expected_pos = np.array(initial_pos) + np.array(initial_vel)
        expected_vel = np.array(initial_vel)

        self.physics_processor.process_generic_movement(game_object)

        np.testing.assert_array_almost_equal(game_object.pos, expected_pos, decimal=5)
        np.testing.assert_array_almost_equal(game_object.vel, expected_vel, decimal=5)

    def test_process_generic_movement_collision_reflect_top_boundary(self):
        obj_radius = 5.0
        # Positioned just below the top boundary, moving up
        # Dish center [100, 100], dish_radius 100. Boundary for center of object is at y = 100 + (100 - 5) = 195
        initial_pos = [100.0, 195.0 - 1.0] # y = 194.0
        initial_vel = [0.0, 2.0] # Moving straight up
        game_object = self._create_test_game_object(pos=initial_pos, vel=initial_vel, radius=obj_radius)
        
        # After collision, object should be on boundary, y_vel reflected
        # Expected position y = 195.0
        # Original velocity was (0, 2). Normal approx (0, -1). Reflected vel = (0,2) - 2 * dot((0,2),(0,-1)) * (0,-1)
        # = (0,2) - 2 * (-2) * (0,-1) = (0,2) + 4 * (0,-1) = (0,2) + (0,-4) = (0,-2)
        
        self.physics_processor.process_generic_movement(game_object)

        # Check velocity is reflected
        self.assertAlmostEqual(game_object.vel[0], 0.0, places=5)
        self.assertLess(game_object.vel[1], 0.0) # Should be negative now
        self.assertAlmostEqual(game_object.vel[1], -initial_vel[1], places=5) # Magnitude should be same

        # Check position is on the boundary
        expected_pos_y = self.mock_env.dish_center[1] + (self.mock_env.dish_radius - game_object.radius)
        self.assertAlmostEqual(game_object.pos[0], initial_pos[0], places=5) # X position shouldn't change much
        self.assertAlmostEqual(game_object.pos[1], expected_pos_y, places=5)
        
        # Verify distance from center after repositioning
        dist_to_center_after = np.linalg.norm(game_object.pos - self.mock_env.dish_center)
        self.assertAlmostEqual(dist_to_center_after, self.mock_env.dish_radius - game_object.radius, places=5)

    def test_process_generic_movement_collision_reflect_diagonal(self):
        obj_radius = 5.0
        # Positioned near top-right corner, moving towards it
        # Boundary for center of object is a circle of radius 95 around [100,100]
        # Initial position: a bit inside the boundary, e.g., at distance 94 from center along diagonal
        offset = (self.mock_env.dish_radius - obj_radius - 1.0) / np.sqrt(2) # 94 / sqrt(2)
        initial_pos = [self.mock_env.dish_center[0] + offset, self.mock_env.dish_center[1] + offset]
        initial_vel = [2.0, 2.0] # Moving towards top-right
        game_object = self._create_test_game_object(pos=initial_pos, vel=initial_vel, radius=obj_radius)

        # Store original velocity for comparison
        original_vel_magnitude = np.linalg.norm(initial_vel)

        self.physics_processor.process_generic_movement(game_object)

        # Check position is on the boundary
        dist_to_center_after = np.linalg.norm(game_object.pos - self.mock_env.dish_center)
        self.assertAlmostEqual(dist_to_center_after, self.mock_env.dish_radius - game_object.radius, places=5)

        # Check velocity is reflected (component-wise signs might not be simple negation)
        # The reflected velocity should be pointing away from the boundary normal
        # Normal at point of impact is roughly (pos - center) / ||pos - center||
        # We expect dot(new_vel, normal_at_impact) < 0 if normal points outwards from center
        normal_at_impact = (game_object.pos - self.mock_env.dish_center) / np.linalg.norm(game_object.pos - self.mock_env.dish_center)
        self.assertTrue(np.dot(game_object.vel, normal_at_impact) < 0, "Velocity not pointing away from boundary")

        # Check magnitude of velocity (assuming elastic collision, speed should be conserved)
        # Note: The reflection formula used v' = v - 2 * dot(v, n) * n preserves magnitude of v if n is a unit vector.
        reflected_vel_magnitude = np.linalg.norm(game_object.vel)
        self.assertAlmostEqual(reflected_vel_magnitude, original_vel_magnitude, places=5, msg="Speed should be conserved after reflection")
        
        # Simple check: both components of velocity should not be positive if moving towards top-right
        self.assertTrue(not (game_object.vel[0] > 0 and game_object.vel[1] > 0), "Velocity not reflected as expected for diagonal collision")

    def _create_test_organism(self, pos, vel, radius=None, max_speed=None):
        org = Organism.__new__(Organism) # Create a bare instance
        org.pos = np.array(pos, dtype=float)
        org.vel = np.array(vel, dtype=float)
        org.radius = float(radius if radius is not None else self.mock_settings['organism']['radius'])
        org.max_speed = float(max_speed if max_speed is not None else self.mock_settings['organism']['max_speed'])
        
        # process_organism_action is called from PhysicsProcessor, 
        # and PhysicsProcessor's self.settings is passed to it.
        # The organism object itself doesn't need its own .settings for this method to be unit-tested.
        # However, if other Organism methods were called, it might.
        # For this specific test, PhysicsProcessor.settings['organism']['max_acceleration'] is used.
        # org.settings = self.mock_settings 
        
        # org.env = self.mock_env # Not used by process_organism_action
        org.object_type = 'organism' # For clarity
        return org

    def test_process_organism_action_apply_acceleration(self):
        initial_vel = [0.0, 0.0]
        organism = self._create_test_organism(pos=[0.0, 0.0], vel=initial_vel)
        action = np.array([1.0, 0.0])
        
        max_acceleration = self.mock_settings['organism']['max_acceleration']
        expected_vel = np.array(initial_vel) + action * max_acceleration

        self.physics_processor.process_organism_action(organism, action)
        
        np.testing.assert_array_almost_equal(organism.vel, expected_vel, decimal=5)
        self.assertTrue(np.linalg.norm(organism.vel) < organism.max_speed) # Ensure no clamping occurred

    def test_process_organism_action_speed_clamping(self):
        custom_max_speed = 2.0
        initial_vel = [1.0, 0.0] # Initial speed is 1.0
        # Organism's max_speed is set during its creation by _create_test_organism
        organism = self._create_test_organism(pos=[0.0, 0.0], vel=initial_vel, max_speed=custom_max_speed)
        
        action = np.array([2.0, 0.0]) # This action will push speed beyond max_speed
        max_acceleration = self.mock_settings['organism']['max_acceleration'] # This is 1.0
        
        # vel = [1,0] + [2,0]*1.0 = [3,0]. Speed = 3.0
        vel_before_clamping = np.array(initial_vel) + action * max_acceleration
        
        self.physics_processor.process_organism_action(organism, action)
        
        # Assert speed is clamped to max_speed
        self.assertAlmostEqual(np.linalg.norm(organism.vel), custom_max_speed, decimal=5)
        
        # Assert direction is preserved
        expected_direction = vel_before_clamping / np.linalg.norm(vel_before_clamping)
        actual_direction = organism.vel / np.linalg.norm(organism.vel)
        np.testing.assert_array_almost_equal(actual_direction, expected_direction, decimal=5)

    def test_process_organism_action_zero_action(self):
        initial_vel = [1.0, 1.0]
        organism = self._create_test_organism(pos=[0.0, 0.0], vel=initial_vel)
        action = np.array([0.0, 0.0])
        
        expected_vel = np.array(initial_vel) # Velocity should not change

        self.physics_processor.process_organism_action(organism, action)
        
        np.testing.assert_array_almost_equal(organism.vel, expected_vel, decimal=5)

    def _create_test_collision_object(self, pos, radius):
        obj = Mock() 
        obj.pos = np.array(pos, dtype=float)
        obj.radius = float(radius)
        return obj

    # Tests for resolve_food_food_collisions
    def test_resolve_food_food_collisions_no_collision(self):
        food1 = self._create_test_collision_object(pos=[0.0, 0.0], radius=5.0)
        food2 = self._create_test_collision_object(pos=[20.0, 0.0], radius=5.0)
        
        initial_pos1 = food1.pos.copy()
        initial_pos2 = food2.pos.copy()

        self.physics_processor.resolve_food_food_collisions([food1, food2])

        np.testing.assert_array_almost_equal(food1.pos, initial_pos1, decimal=5)
        np.testing.assert_array_almost_equal(food2.pos, initial_pos2, decimal=5)

    def test_resolve_food_food_collisions_simple_overlap(self):
        food1 = self._create_test_collision_object(pos=[50.0, 50.0], radius=5.0) # Center at 50
        food2 = self._create_test_collision_object(pos=[55.0, 50.0], radius=5.0) # Center at 55, edge at 50. Overlap is 5.
        
        # Total radius = 10. Distance = 5. Overlap = 10 - 5 = 5.
        # Each moves by overlap / 2 = 2.5
        # food1 moves from 50 to 50 - 2.5 = 47.5
        # food2 moves from 55 to 55 + 2.5 = 57.5
        expected_pos1 = [47.5, 50.0]
        expected_pos2 = [57.5, 50.0]

        self.physics_processor.resolve_food_food_collisions([food1, food2])

        np.testing.assert_array_almost_equal(food1.pos, expected_pos1, decimal=5)
        np.testing.assert_array_almost_equal(food2.pos, expected_pos2, decimal=5)

    def test_resolve_food_food_collisions_multiple_objects_some_overlap(self):
        food1 = self._create_test_collision_object(pos=[50.0, 50.0], radius=5.0)
        food2 = self._create_test_collision_object(pos=[55.0, 50.0], radius=5.0) # Overlaps with food1
        food3 = self._create_test_collision_object(pos=[100.0, 100.0], radius=5.0) # No overlap

        expected_pos1 = [47.5, 50.0]
        expected_pos2 = [57.5, 50.0]
        initial_pos3 = food3.pos.copy()

        self.physics_processor.resolve_food_food_collisions([food1, food2, food3])

        np.testing.assert_array_almost_equal(food1.pos, expected_pos1, decimal=5)
        np.testing.assert_array_almost_equal(food2.pos, expected_pos2, decimal=5)
        np.testing.assert_array_almost_equal(food3.pos, initial_pos3, decimal=5)

    # Tests for resolve_food_obstacle_collisions
    def test_resolve_food_obstacle_collisions_no_collision(self):
        food = self._create_test_collision_object(pos=[0.0, 0.0], radius=5.0)
        obstacle = self._create_test_collision_object(pos=[20.0, 0.0], radius=5.0)
        
        initial_food_pos = food.pos.copy()
        initial_obstacle_pos = obstacle.pos.copy()

        self.physics_processor.resolve_food_obstacle_collisions([food], [obstacle])

        np.testing.assert_array_almost_equal(food.pos, initial_food_pos, decimal=5)
        np.testing.assert_array_almost_equal(obstacle.pos, initial_obstacle_pos, decimal=5) # Obstacle should not move

    def test_resolve_food_obstacle_collisions_simple_overlap(self):
        food = self._create_test_collision_object(pos=[50.0, 50.0], radius=5.0) # Edge at 55
        obstacle = self._create_test_collision_object(pos=[53.0, 50.0], radius=5.0) # Edge at 48
        
        # Distance = 3. Sum of radii = 10. Overlap = 10 - 3 = 7.
        # Food moves by the full overlap.
        # Direction from obstacle to food: food.pos - obstacle.pos = [50-53, 50-50] = [-3, 0]. Normalized = [-1, 0]
        # Food moves from 50.0 to 50.0 + (-1 * 7) = 43.0
        expected_food_pos = [43.0, 50.0]
        initial_obstacle_pos = obstacle.pos.copy()

        self.physics_processor.resolve_food_obstacle_collisions([food], [obstacle])

        np.testing.assert_array_almost_equal(food.pos, expected_food_pos, decimal=5)
        np.testing.assert_array_almost_equal(obstacle.pos, initial_obstacle_pos, decimal=5) # Obstacle should not move

    def test_resolve_food_obstacle_collisions_food_completely_inside_obstacle_centered(self):
        food = self._create_test_collision_object(pos=[50.0, 50.0], radius=2.0)
        obstacle = self._create_test_collision_object(pos=[50.0, 50.0], radius=10.0)
        
        initial_food_pos = food.pos.copy() # Expect food not to move due to (0,0) direction vector
        initial_obstacle_pos = obstacle.pos.copy()

        self.physics_processor.resolve_food_obstacle_collisions([food], [obstacle])

        # Current implementation: if dist is 0, direction is (0,0), so food doesn't move.
        np.testing.assert_array_almost_equal(food.pos, initial_food_pos, decimal=5)
        np.testing.assert_array_almost_equal(obstacle.pos, initial_obstacle_pos, decimal=5)

    def test_resolve_food_obstacle_collisions_food_inside_obstacle_offset(self):
        # Test when food is inside but not centered, to ensure it's pushed out.
        food_pos = [51.0, 50.0]
        food_radius = 2.0
        obstacle_pos = [50.0, 50.0]
        obstacle_radius = 10.0
        
        food = self._create_test_collision_object(pos=food_pos, radius=food_radius)
        obstacle = self._create_test_collision_object(pos=obstacle_pos, radius=obstacle_radius)

        initial_obstacle_pos = obstacle.pos.copy()
        
        # dist = 1.0. Sum of radii = 12.0. Overlap = 12.0 - 1.0 = 11.0.
        # Direction from obstacle to food: [51-50, 50-50] = [1,0]. Normalized = [1,0]
        # Food moves from 51.0 to 51.0 + (1.0 * 11.0) = 62.0
        # Expected pos: [62.0, 50.0]
        expected_food_pos = [62.0, 50.0]

        self.physics_processor.resolve_food_obstacle_collisions([food], [obstacle])

        np.testing.assert_array_almost_equal(food.pos, expected_food_pos, decimal=5)
        np.testing.assert_array_almost_equal(obstacle.pos, initial_obstacle_pos, decimal=5)

    # Tests for resolve_generic_object_collision
    def test_resolve_generic_object_collision_no_collision(self):
        obj1 = self._create_test_collision_object(pos=[0.0, 0.0], radius=5.0)
        obj2 = self._create_test_collision_object(pos=[20.0, 0.0], radius=5.0)
        
        initial_pos1 = obj1.pos.copy()
        initial_pos2 = obj2.pos.copy()

        self.physics_processor.resolve_generic_object_collision(obj1, obj2)

        np.testing.assert_array_almost_equal(obj1.pos, initial_pos1, decimal=5)
        np.testing.assert_array_almost_equal(obj2.pos, initial_pos2, decimal=5)

    def test_resolve_generic_object_collision_same_size_objects(self):
        # obj1 at [50, 50] (radius 5), obj2 at [55, 50] (radius 5). Overlap = 5.
        obj1 = self._create_test_collision_object(pos=[50.0, 50.0], radius=5.0)
        obj2 = self._create_test_collision_object(pos=[55.0, 50.0], radius=5.0)
        
        # displacement_obj1 = 5 * (5 / 10) = 2.5
        # displacement_obj2 = 5 * (5 / 10) = 2.5
        # Direction = obj1.pos - obj2.pos = [-5,0], normalized = [-1,0]
        # Expected obj1.pos = [50,50] + [-1,0]*2.5 = [47.5, 50]
        # Expected obj2.pos = [55,50] - [-1,0]*2.5 = [57.5, 50]
        expected_pos1 = [47.5, 50.0]
        expected_pos2 = [57.5, 50.0]

        self.physics_processor.resolve_generic_object_collision(obj1, obj2)

        np.testing.assert_array_almost_equal(obj1.pos, expected_pos1, decimal=5)
        np.testing.assert_array_almost_equal(obj2.pos, expected_pos2, decimal=5)

    def test_resolve_generic_object_collision_small_and_large_objects(self):
        # obj1 (small, r=2) at [50, 50], obj2 (large, r=8) at [58, 50]. Overlap = 2.
        obj1 = self._create_test_collision_object(pos=[50.0, 50.0], radius=2.0)
        obj2 = self._create_test_collision_object(pos=[58.0, 50.0], radius=8.0)

        # displacement_obj1 = 2 * (8 / 10) = 1.6
        # displacement_obj2 = 2 * (2 / 10) = 0.4
        # Direction = obj1.pos - obj2.pos = [-8,0], normalized = [-1,0]
        # Expected obj1.pos = [50,50] + [-1,0]*1.6 = [48.4, 50]
        # Expected obj2.pos = [58,50] - [-1,0]*0.4 = [58.4, 50]
        expected_pos1 = [48.4, 50.0]
        expected_pos2 = [58.4, 50.0]

        self.physics_processor.resolve_generic_object_collision(obj1, obj2)

        np.testing.assert_array_almost_equal(obj1.pos, expected_pos1, decimal=5)
        np.testing.assert_array_almost_equal(obj2.pos, expected_pos2, decimal=5)

    def test_resolve_generic_object_collision_perfectly_overlapping_centers(self):
        # obj1 (r=5) and obj2 (r=3) both at [50, 50]. Overlap = 8.
        obj1 = self._create_test_collision_object(pos=[50.0, 50.0], radius=5.0)
        obj2 = self._create_test_collision_object(pos=[50.0, 50.0], radius=3.0)
        
        # displacement_obj1 = 8 * (3 / 8) = 3
        # displacement_obj2 = 8 * (5 / 8) = 5
        # Direction defaults to [1,0]
        # Expected obj1.pos = [50,50] + [1,0]*3 = [53, 50]
        # Expected obj2.pos = [50,50] - [1,0]*5 = [45, 50]
        expected_pos1 = [53.0, 50.0]
        expected_pos2 = [45.0, 50.0]

        self.physics_processor.resolve_generic_object_collision(obj1, obj2)

        np.testing.assert_array_almost_equal(obj1.pos, expected_pos1, decimal=5)
        np.testing.assert_array_almost_equal(obj2.pos, expected_pos2, decimal=5)

    def test_resolve_generic_object_collision_touching_no_overlap(self):
        # obj1 (r=5) at [50,50], obj2 (r=5) at [60,50]. Overlap = 0.
        obj1 = self._create_test_collision_object(pos=[50.0, 50.0], radius=5.0)
        obj2 = self._create_test_collision_object(pos=[60.0, 50.0], radius=5.0)

        initial_pos1 = obj1.pos.copy()
        initial_pos2 = obj2.pos.copy()

        self.physics_processor.resolve_generic_object_collision(obj1, obj2)

        # Overlap is 0, so if overlap > 1e-6 is false, no change.
        # If overlap is positive but very small (e.g. 1e-7 due to float precision),
        # then a tiny change might happen. The test assumes overlap <= 1e-6 means no change.
        np.testing.assert_array_almost_equal(obj1.pos, initial_pos1, decimal=5)
        np.testing.assert_array_almost_equal(obj2.pos, initial_pos2, decimal=5)


if __name__ == '__main__':
    unittest.main()
