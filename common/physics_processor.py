import numpy as np

class PhysicsProcessor:
    def __init__(self, env, settings):
        self.env = env
        self.settings = settings

    def process_generic_movement(self, game_object):
        # Update position
        game_object.pos += game_object.vel

        # Calculate distance to center
        dist_to_center = np.linalg.norm(game_object.pos - self.env.dish_center)

        # Check for collision
        if dist_to_center > self.env.dish_radius - game_object.radius:
            # Calculate normal vector (pointing from object towards dish center)
            # Note: The original GameObject.move had normal = (self.env.dish_center - self.pos) / (dist_to_center + 1e-6)
            # which is already pointing inwards (from object to center if pos is outside, or from center to object if pos is inside).
            # For reflection, we want the normal of the surface, which points from the center outwards to the object.
            # However, the reflection formula v' = v - 2 * dot(v, n) * n assumes n is the surface normal.
            # If we use a normal pointing inwards (from object to center), the reflection still works out correctly
            # because the repositioning step will place the object on the boundary.
            # Let's stick to the logic similar to GameObject.move for the normal calculation direction for now.
            normal = (self.env.dish_center - game_object.pos) / (dist_to_center + 1e-6) # Normal pointing inwards

            # Reflect velocity
            dot_product = np.dot(game_object.vel, normal)
            game_object.vel = game_object.vel - 2 * dot_product * normal

            # Reposition to be exactly on the boundary
            # Direction from center to object
            direction_from_center = (game_object.pos - self.env.dish_center) / (dist_to_center + 1e-6)
            game_object.pos = self.env.dish_center + direction_from_center * (self.env.dish_radius - game_object.radius)

    def process_organism_action(self, organism, action):
        # Calculate acceleration
        acceleration = np.array(action) * self.settings['organism']['max_acceleration']

        # Update organism's velocity
        organism.vel += acceleration

        # Calculate current speed
        speed = np.linalg.norm(organism.vel)

        # Cap speed if it exceeds max_speed
        if speed > organism.max_speed and speed > 0:
            organism.vel = organism.vel / speed * organism.max_speed

    def resolve_food_food_collisions(self, foods):
        for i, food1 in enumerate(foods):
            for food2 in foods[i+1:]: # Starts from i+1 to get unique pairs
                dist = np.linalg.norm(food1.pos - food2.pos)
                if dist < food1.radius + food2.radius and dist > 0: # Ensure dist > 0
                    direction = (food1.pos - food2.pos) / (dist + 1e-6) # Added epsilon for safety
                    overlap = food1.radius + food2.radius - dist
                    food1.pos += direction * (overlap / 2)
                    food2.pos -= direction * (overlap / 2)

    def resolve_food_obstacle_collisions(self, foods, obstacles):
        for food in foods:
            for obstacle in obstacles:
                dist = np.linalg.norm(food.pos - obstacle.pos)
                if dist < food.radius + obstacle.radius:
                    direction = (food.pos - obstacle.pos) / (dist + 1e-6) # Epsilon for safety
                    overlap = food.radius + obstacle.radius - dist
                    food.pos += direction * overlap

    def resolve_generic_object_collision(self, obj1, obj2):
        dist = np.linalg.norm(obj1.pos - obj2.pos)
        total_radii = obj1.radius + obj2.radius
        overlap = total_radii - dist

        if overlap > 1e-6:
            direction = obj1.pos - obj2.pos

            # Handle cases where objects are at the same position or very close
            if np.linalg.norm(direction) < 1e-6:
                direction = np.array([1.0, 0.0]) # Arbitrary direction for separation
            else:
                direction = direction / np.linalg.norm(direction) # Normalize

            # Ensure total_radii is not zero to prevent division by zero error.
            # This should generally be true if radii are positive.
            if total_radii > 1e-6: # Added safety for total_radii
                displacement1 = overlap * (obj2.radius / total_radii)
                displacement2 = overlap * (obj1.radius / total_radii)
            else:
                # If total_radii is effectively zero, split overlap equally or handle as an error.
                # For now, splitting equally if this unlikely case occurs.
                displacement1 = overlap / 2
                displacement2 = overlap / 2

            obj1.pos = obj1.pos + direction * displacement1
            obj2.pos = obj2.pos - direction * displacement2

    def apply_surface_friction(self, obj, delta_time):
        # Retrieve gravitational constant g from settings, with a default value
        physics_constants = self.settings.get('physics_constants', {})
        g = physics_constants.get('g', 9.8)

        speed = np.linalg.norm(obj.vel)

        # If the object is practically stationary, do nothing
        if speed < 1e-6:
            return

        # Calculate deceleration due to friction
        # Assumes obj has a 'friction_coefficient' attribute
        deceleration = obj.friction_coefficient * g

        # Calculate the potential change in speed in this time step
        delta_speed = deceleration * delta_time

        # If friction would stop the object or reverse its velocity
        if delta_speed >= speed:
            obj.vel = np.array([0.0, 0.0])
        else:
            # Friction slows the object but doesn't stop it
            new_speed = speed - delta_speed
            obj.vel = (obj.vel / speed) * new_speed # Maintain direction, reduce magnitude

    def resolve_billiard_ball_collision(self, obj1, obj2, is_obstacle_food_pair=False):
        # Part 1: Positional Correction (Prevent Overlap)
        initial_dist = np.linalg.norm(obj1.pos - obj2.pos)
        total_radii = obj1.radius + obj2.radius
        overlap = total_radii - initial_dist

        original_obj1_vel = obj1.vel.copy() # Store original velocity for obstacle in obstacle-food pair

        if overlap > 1e-6:
            direction = obj1.pos - obj2.pos
            norm_direction = np.linalg.norm(direction)

            if norm_direction < 1e-6:
                direction_normalized = np.array([1.0, 0.0]) # Arbitrary direction for separation
            else:
                direction_normalized = direction / norm_direction

            m1 = obj1.mass
            m2 = obj2.mass
            total_mass = m1 + m2

            if total_mass < 1e-6: # Effectively massless pair
                displacement1 = overlap / 2.0
                displacement2 = overlap / 2.0
            else:
                displacement1 = overlap * (m2 / total_mass)
                displacement2 = overlap * (m1 / total_mass)

            obj1.pos = obj1.pos + direction_normalized * displacement1
            obj2.pos = obj2.pos - direction_normalized * displacement2

            # Part 2: Velocity Update (Elastic Collision)
            # Normal vector based on corrected positions (or direction_normalized can be used)
            # If objects were perfectly overlapping and then separated, direction_normalized is the normal.
            # If they were already separated, direction_normalized is still the line between centers.
            normal_vec = direction_normalized # This is already normalized and points from obj2 to obj1

            # Tangent vector
            tangent_vec = np.array([-normal_vec[1], normal_vec[0]])

            # Project velocities onto normal and tangent vectors
            v1n = np.dot(obj1.vel, normal_vec)
            v1t = np.dot(obj1.vel, tangent_vec)
            v2n = np.dot(obj2.vel, normal_vec)
            v2t = np.dot(obj2.vel, tangent_vec)

            # Calculate new normal velocities (1D elastic collision formula with restitution)
            combined_elasticity = (obj1.elasticity + obj2.elasticity) / 2.0

            if total_mass < 1e-6: # Massless pair
                # Simplification: normal velocities are unchanged or swap if e=1
                # For now, unchanged as per plan if total_mass is zero.
                # If one is massive and other is not, the formula below actually handles it.
                # This case is truly for when m1 and m2 are both near zero.
                new_v1n_final = v1n
                new_v2n_final = v2n
            else:
                new_v1n_final = (m1*v1n + m2*v2n - m2*(v1n - v2n)*combined_elasticity) / total_mass
                new_v2n_final = (m1*v1n + m2*v2n - m1*(v2n - v1n)*combined_elasticity) / total_mass

            # Convert new normal velocities back to vector form and add tangent components
            new_v1_vec = new_v1n_final * normal_vec + v1t * tangent_vec
            new_v2_vec = new_v2n_final * normal_vec + v2t * tangent_vec

            # Apply velocities
            if is_obstacle_food_pair:
                # obj1 is Obstacle, obj2 is Food
                obj1.vel = original_obj1_vel # Obstacle velocity remains unchanged
                obj2.vel = new_v2_vec
            else:
                obj1.vel = new_v1_vec
                obj2.vel = new_v2_vec
        # else: if overlap <= 1e-6, no positional or velocity update from collision is needed.

    def resolve_organism_positional_collision(self, organism, other_object):
        dist = np.linalg.norm(organism.pos - other_object.pos)
        total_radii = organism.radius + other_object.radius
        overlap = total_radii - dist

        if overlap > 1e-6:
            direction = organism.pos - other_object.pos
            norm_direction_val = np.linalg.norm(direction)

            if norm_direction_val < 1e-6:
                direction_normalized = np.array([1.0, 0.0]) # Arbitrary separation axis
            else:
                direction_normalized = direction / norm_direction_val

            m_organism = organism.mass
            m_other = other_object.mass
            total_mass = m_organism + m_other

            if total_mass < 1e-6: # Effectively massless pair
                displacement_organism = overlap / 2.0
                displacement_other = overlap / 2.0
            else: # Standard mass-based displacement
                displacement_organism = overlap * (m_other / total_mass)
                displacement_other = overlap * (m_organism / total_mass)

            organism.pos = organism.pos + direction_normalized * displacement_organism
            other_object.pos = other_object.pos - direction_normalized * displacement_other

        # This method explicitly does not modify velocities.

    def handle_dish_boundary(self, obj):
        dist_to_center = np.linalg.norm(obj.pos - self.env.dish_center)

        # Ensure obj has a radius attribute, defaulting to 0 if not (though objects should have it)
        obj_radius = obj.radius if hasattr(obj, 'radius') else 0.0

        if dist_to_center > self.env.dish_radius - obj_radius:
            # Normal vector of the collision (points from dish center towards the object, normalized)
            # This vector points outwards from the dish center to the object's center.
            # This is the normal of the surface FROM the object's perspective.
            # For the reflection formula v_new = v - 2 * dot(v, n) * n, 'n' should be the normal of the surface hit.
            # So, collision_normal should point from the object towards the center, or from boundary point to center.
            # The provided example uses (obj.pos - self.env.dish_center), which is outward from center.
            # Let's use this for consistency with the example.

            collision_normal = (obj.pos - self.env.dish_center) / (dist_to_center + 1e-6)

            # Reflect velocity: v_new = v - 2 * dot(v, n) * n
            # Ensure obj.vel is a numpy array
            if not isinstance(obj.vel, np.ndarray):
                obj.vel = np.array(obj.vel, dtype=float)

            obj.vel = obj.vel - 2 * np.dot(obj.vel, collision_normal) * collision_normal

            # Correct position to be on the boundary
            # The object should be placed exactly at (dish_radius - obj.radius) from the center
            # along the collision_normal direction (which is already the normalized direction from center to object)
            obj.pos = self.env.dish_center + collision_normal * (self.env.dish_radius - obj_radius)
