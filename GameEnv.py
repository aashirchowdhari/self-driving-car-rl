# Import pygame library for game development
import pygame
# Import math for mathematical operations
import math
# Import Wall class from Walls module
from Walls import Wall
# Import getWalls function from Walls module
from Walls import getWalls
# Import Goal class from Goals module
from Goals import Goal
# Import getGoals function from Goals module
from Goals import getGoals

# Define constant for reward when reaching a goal
GOALREWARD = 1
# Define constant for reward per step (life reward)
LIFE_REWARD = 0
# Define constant for penalty when crashing
PENALTY = -1

# Function to calculate Euclidean distance between two points
def distance(pt1, pt2):
    # Return the square root of the sum of squared differences
    return(((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5)

# Function to rotate a point around an origin by a given angle
def rotate(origin, point, angle):
    # Calculate new x-coordinate after rotation
    qx = origin.x + math.cos(angle) * (point.x - origin.x) - math.sin(angle) * (point.y - origin.y)
    # Calculate new y-coordinate after rotation
    qy = origin.y + math.sin(angle) * (point.x - origin.x) + math.cos(angle) * (point.y - origin.y)
    # Create and return a new point with rotated coordinates
    q = myPoint(qx, qy)
    return q

# Function to rotate a rectangle (defined by four points) around its center
def rotateRect(pt1, pt2, pt3, pt4, angle):
    # Calculate the center point of the rectangle
    pt_center = myPoint((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)
    # Rotate each corner point around the center
    pt1 = rotate(pt_center, pt1, angle)
    pt2 = rotate(pt_center, pt2, angle)
    pt3 = rotate(pt_center, pt3, angle)
    pt4 = rotate(pt_center, pt4, angle)
    # Return the rotated points
    return pt1, pt2, pt3, pt4

# Class to represent a point with x and y coordinates
class myPoint:
    # Initialize point with x and y coordinates
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
# Class to represent a line segment between two points
class myLine:
    # Initialize line with two points
    def __init__(self, pt1, pt2):
        self.pt1 = myPoint(pt1.x, pt1.y)
        self.pt2 = myPoint(pt2.x, pt2.y)

# Class to represent a ray for raycasting
class Ray:
    # Initialize ray with origin (x, y) and angle
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    # Method to cast ray against a wall and find intersection
    def cast(self, wall):
        # Wall endpoints
        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        # Create a vector for the ray direction (length 1000)
        vec = rotate(myPoint(0,0), myPoint(0,-1000), self.angle)
        
        # Ray endpoints
        x3 = self.x
        y3 = self.y
        x4 = self.x + vec.x
        y4 = self.y + vec.y

        # Calculate denominator for line intersection formula
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
        # Check if lines are parallel
        if den == 0:
            den = 0
        else:
            # Calculate intersection parameters t and u
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            # Check if intersection is within line segments
            if t > 0 and t < 1 and u < 1 and u > 0:
                # Calculate and return intersection point
                pt = myPoint(math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1)))
                return pt

# Class to represent the car in the racing game
class Car:
    # Initialize car with starting position (x, y)
    def __init__(self, x, y):
        self.pt = myPoint(x, y)
        self.x = x
        self.y = y
        # Set car dimensions
        self.width = 14
        self.height = 30

        # Initialize points counter
        self.points = 0

        # Load and set up car image
        self.original_image = pygame.image.load("car.png").convert()
        self.image = self.original_image  # Reference to rotated image
        self.image.set_colorkey((0,0,0))  # Set black as transparent
        self.rect = self.image.get_rect().move(self.x, self.y)

        # Initialize car angle (facing up)
        self.angle = math.radians(180)
        self.soll_angle = self.angle

        # Initialize velocity parameters
        self.dvel = 1
        self.vel = 0
        self.velX = 0
        self.velY = 0
        self.maxvel = 15  # Maximum velocity

        # Reinitialize angle (redundant, could be removed)
        self.angle = math.radians(180)
        self.soll_angle = self.angle

        # Define car corners
        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        # Initialize rotated corner points
        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

        # Initialize list to store raycast distances
        self.distances = []

    # Method to perform actions based on input choice
    def action(self, choice):
        # No action
        if choice == 0:
            pass
        # Accelerate forward
        elif choice == 1:
            self.accelerate(self.dvel)
        # Accelerate and turn left
        elif choice == 8:
            self.accelerate(self.dvel)
            self.turn(1)
        # Accelerate and turn right
        elif choice == 7:
            self.accelerate(self.dvel)
            self.turn(-1)
        # Accelerate backward
        elif choice == 4:
            self.accelerate(-self.dvel)
        # Accelerate backward and turn left
        elif choice == 5:
            self.accelerate(-self.dvel)
            self.turn(1)
        # Accelerate backward and turn right
        elif choice == 6:
            self.accelerate(-self.dvel)
            self.turn(-1)
        # Turn left
        elif choice == 3:
            self.turn(1)
        # Turn right
        elif choice == 2:
            self.turn(-1)
        pass
    
    # Method to change velocity
    def accelerate(self, dvel):
        # Double the acceleration for effect
        dvel = dvel * 2
        # Update velocity
        self.vel = self.vel + dvel

        # Cap velocity at maximum
        if self.vel > self.maxvel:
            self.vel = self.maxvel
        # Cap velocity at negative maximum
        if self.vel < -self.maxvel:
            self.vel = -self.maxvel

    # Method to change target angle
    def turn(self, dir):
        # Update target angle by 15 degrees in specified direction
        self.soll_angle = self.soll_angle + dir * math.radians(15)
    
    # Method to update car position and rotation
    def update(self):
        # Commented-out drifting code for smoother turning
        # if(self.soll_angle > self.angle):
        #     if(self.soll_angle > self.angle + math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)):
        #         self.angle = self.angle + math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)
        #     else:
        #         self.angle = self.soll_angle
        # if(self.soll_angle < self.angle):
        #     if(self.soll_angle < self.angle - math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)):
        #         self.angle = self.angle - math.radians(10) * self.maxvel / ((self.velX**2 + self.velY**2)**0.5 + 1)
        #     else:
        #         self.angle = self.soll_angle
        
        # Set current angle to target angle
        self.angle = self.soll_angle

        # Calculate velocity components based on angle
        vec_temp = rotate(myPoint(0,0), myPoint(0, self.vel), self.angle)
        self.velX, self.velY = vec_temp.x, vec_temp.y

        # Update car position
        self.x = self.x + self.velX
        self.y = self.y + self.velY

        # Update rectangle center
        self.rect.center = self.x, self.y

        # Update corner points with velocity
        self.pt1 = myPoint(self.pt1.x + self.velX, self.pt1.y + self.velY)
        self.pt2 = myPoint(self.pt2.x + self.velX, self.pt2.y + self.velY)
        self.pt3 = myPoint(self.pt3.x + self.velX, self.pt3.y + self.velY)
        self.pt4 = myPoint(self.pt4.x + self.velX, self.pt4.y + self.velY)

        # Rotate car corners
        self.p1, self.p2, self.p3, self.p4 = rotateRect(self.pt1, self.pt2, self.pt3, self.pt4, self.soll_angle)

        # Rotate car image and update rectangle
        self.image = pygame.transform.rotate(self.original_image, 90 - self.soll_angle * 180 / math.pi)
        x, y = self.rect.center  # Save current center
        self.rect = self.image.get_rect()  # Get new rectangle
        self.rect.center = (x, y)  # Restore center

    # Method to cast rays and get distances to walls
    def cast(self, walls):
        # Define rays at various angles relative to car
        ray1 = Ray(self.x, self.y, self.soll_angle)
        ray2 = Ray(self.x, self.y, self.soll_angle - math.radians(30))
        ray3 = Ray(self.x, self.y, self.soll_angle + math.radians(30))
        ray4 = Ray(self.x, self.y, self.soll_angle + math.radians(45))
        ray5 = Ray(self.x, self.y, self.soll_angle - math.radians(45))
        ray6 = Ray(self.x, self.y, self.soll_angle + math.radians(90))
        ray7 = Ray(self.x, self.y, self.soll_angle - math.radians(90))
        ray8 = Ray(self.x, self.y, self.soll_angle + math.radians(180))
        ray9 = Ray(self.x, self.y, self.soll_angle + math.radians(10))
        ray10 = Ray(self.x, self.y, self.soll_angle - math.radians(10))
        ray11 = Ray(self.x, self.y, self.soll_angle + math.radians(135))
        ray12 = Ray(self.x, self.y, self.soll_angle - math.radians(135))
        ray13 = Ray(self.x, self.y, self.soll_angle + math.radians(20))
        ray14 = Ray(self.x, self.y, self.soll_angle - math.radians(20))
        ray15 = Ray(self.p1.x, self.p1.y, self.soll_angle + math.radians(90))
        ray16 = Ray(self.p2.x, self.p2.y, self.soll_angle - math.radians(90))
        ray17 = Ray(self.p1.x, self.p1.y, self.soll_angle + math.radians(0))
        ray18 = Ray(self.p2.x, self.p2.y, self.soll_angle - math.radians(0))

        # Store rays in a list
        self.rays = []
        self.rays.append(ray1)
        self.rays.append(ray2)
        self.rays.append(ray3)
        self.rays.append(ray4)
        self.rays.append(ray5)
        self.rays.append(ray6)
        self.rays.append(ray7)
        self.rays.append(ray8)
        self.rays.append(ray9)
        self.rays.append(ray10)
        self.rays.append(ray11)
        self.rays.append(ray12)
        self.rays.append(ray13)
        self.rays.append(ray14)
        self.rays.append(ray15)
        self.rays.append(ray16)
        self.rays.append(ray17)
        self.rays.append(ray18)

        # Initialize lists for observations and closest points
        observations = []
        self.closestRays = []

        # Cast each ray against all walls
        for ray in self.rays:
            closest = None  # Closest intersection point
            record = math.inf  # Shortest distance
            for wall in walls:
                pt = ray.cast(wall)
                if pt:
                    dist = distance(myPoint(self.x, self.y), pt)
                    if dist < record:
                        record = dist
                        closest = pt

            if closest: 
                # Store closest point and distance
                self.closestRays.append(closest)
                observations.append(record)
            else:
                # Use large distance if no intersection
                observations.append(1000)

        # Normalize distances (0 far, 1 close)
        for i in range(len(observations)):
            observations[i] = ((1000 - observations[i]) / 1000)

        # Append normalized velocity
        observations.append(self.vel / self.maxvel)
        return observations

    # Method to check collision with a wall
    def collision(self, wall):
        # Define car edges
        line1 = myLine(self.p1, self.p2)
        line2 = myLine(self.p2, self.p3)
        line3 = myLine(self.p3, self.p4)
        line4 = myLine(self.p4, self.p1)

        # Wall endpoints
        x1 = wall.x1 
        y1 = wall.y1
        x2 = wall.x2
        y2 = wall.y2

        # List of car edges
        lines = []
        lines.append(line1)
        lines.append(line2)
        lines.append(line3)
        lines.append(line4)

        # Check each edge against the wall
        for li in lines:
            x3 = li.pt1.x
            y3 = li.pt1.y
            x4 = li.pt2.x
            y4 = li.pt2.y

            # Calculate intersection
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            
            if den == 0:
                den = 0
            else:
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

                # Check if intersection is within segments
                if t > 0 and t < 1 and u < 1 and u > 0:
                    return True
        
        return False
    
    # Method to check if car scores by crossing a goal
    def score(self, goal):
        # Define line from car center in direction of movement
        line1 = myLine(self.p1, self.p3)
        vec = rotate(myPoint(0,0), myPoint(0,-50), self.angle)
        line1 = myLine(myPoint(self.x, self.y), myPoint(self.x + vec.x, self.y + vec.y))

        # Goal endpoints
        x1 = goal.x1 
        y1 = goal.y1
        x2 = goal.x2
        y2 = goal.y2
            
        # Car line endpoints
        x3 = line1.pt1.x
        y3 = line1.pt1.y
        x4 = line1.pt2.x
        y4 = line1.pt2.y

        # Calculate intersection
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if den == 0:
            den = 0
        else:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            # Check if intersection is within segments
            if t > 0 and t < 1 and u < 1 and u > 0:
                pt = math.floor(x1 + t * (x2 - x1)), math.floor(y1 + t * (y2 - y1))
                # Check distance to intersection
                d = distance(myPoint(self.x, self.y), myPoint(pt[0], pt[1]))
                if d < 20:
                    # Increment points and return True
                    self.points += GOALREWARD
                    return True

        return False

    # Method to reset car to initial state
    def reset(self):
        # Reset position
        self.x = 50
        self.y = 300
        # Reset velocities
        self.velX = 0
        self.velY = 0
        self.vel = 0
        # Reset angle
        self.angle = math.radians(180)
        self.soll_angle = self.angle
        # Reset points
        self.points = 0

        # Reset corner points
        self.pt1 = myPoint(self.pt.x - self.width / 2, self.pt.y - self.height / 2)
        self.pt2 = myPoint(self.pt.x + self.width / 2, self.pt.y - self.height / 2)
        self.pt3 = myPoint(self.pt.x + self.width / 2, self.pt.y + self.height / 2)
        self.pt4 = myPoint(self.pt.x - self.width / 2, self.pt.y + self.height / 2)

        # Reset rotated points
        self.p1 = self.pt1
        self.p2 = self.pt2
        self.p3 = self.pt3
        self.p4 = self.pt4

    # Method to draw car on screen
    def draw(self, win):
        win.blit(self.image, self.rect)

# Class to represent the racing environment
class RacingEnv:
    # Initialize the environment
    def __init__(self):
        # Initialize pygame
        pygame.init()
        # Set up font for rendering text
        self.font = pygame.font.Font(pygame.font.get_default_font(), 36)

        # Set frames per second
        self.fps = 120
        # Set window dimensions
        self.width = 1000
        self.height = 600
        # Initialize history list
        self.history = []

        # Create display window
        self.screen = pygame.display.set_mode((self.width, self.height))
        # Set window title
        pygame.display.set_caption("RACING DQN")
        # Clear screen
        self.screen.fill((0,0,0))
        # Load background image
        self.back_image = pygame.image.load("track.png").convert()
        self.back_rect = self.back_image.get_rect().move(0, 0)
        # Initialize action and observation spaces (not set)
        self.action_space = None
        self.observation_space = None
        # Initialize game reward
        self.game_reward = 0
        # Initialize score
        self.score = 0
 
        # Reset environment
        self.reset()

    # Method to reset environment
    def reset(self):
        # Clear screen
        self.screen.fill((0, 0, 0))
        # Create new car
        self.car = Car(50, 300)
        # Get walls from Walls module
        self.walls = getWalls()
        # Get goals from Goals module
        self.goals = getGoals()
        # Reset game reward
        self.game_reward = 0

    # Method to perform one step in the environment
    def step(self, action):
        # Initialize done flag
        done = False
        # Perform car action
        self.car.action(action)
        # Update car state
        self.car.update()
        # Set default reward
        reward = LIFE_REWARD

        # Check goals
        index = 1
        for goal in self.goals:
            # Wrap index around
            if index > len(self.goals):
                index = 1
            # Check if car scores on active goal
            if goal.isactiv:
                if self.car.score(goal):
                    goal.isactiv = False
                    self.goals[index-2].isactiv = True
                    reward += GOALREWARD
            index = index + 1

        # Check for collisions with walls
        for wall in self.walls:
            if self.car.collision(wall):
                reward += PENALTY
                done = True

        # Get new state from raycasting
        new_state = self.car.cast(self.walls)
        # Set state to None if done
        if done:
            new_state = None

        return new_state, reward, done

    # Method to render the environment
    def render(self, action):
        # Flags to control rendering of walls, goals, and rays
        DRAW_WALLS = False
        DRAW_GOALS = False
        DRAW_RAYS = False

        # Delay for frame timing
        pygame.time.delay(10)

        # Initialize clock for FPS control
        self.clock = pygame.time.Clock()
        # Clear screen
        self.screen.fill((0, 0, 0))

        # Draw background
        self.screen.blit(self.back_image, self.back_rect)

        # Draw walls if enabled
        if DRAW_WALLS:
            for wall in self.walls:
                wall.draw(self.screen)
        
        # Draw goals if enabled
        if DRAW_GOALS:
            for goal in self.goals:
                goal.draw(self.screen)
                if goal.isactiv:
                    goal.draw(self.screen)
        
        # Draw car
        self.car.draw(self.screen)

        # Draw rays if enabled
        if DRAW_RAYS:
            i = 0
            for pt in self.car.closestRays:
                # Draw intersection points
                pygame.draw.circle(self.screen, (0,0,255), (pt.x, pt.y), 5)
                i += 1
                # Draw rays from car center
                if i < 15:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.x, self.car.y), (pt.x, pt.y), 1)
                # Draw rays from car front midpoint
                elif i >= 15 and i < 17:
                    pygame.draw.line(self.screen, (255,255,255), ((self.car.p1.x + self.car.p2.x)/2, (self.car.p1.y + self.car.p2.y)/2), (pt.x, pt.y), 1)
                # Draw ray from p1
                elif i == 17:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p1.x, self.car.p1.y), (pt.x, pt.y), 1)
                # Draw ray from p2
                else:
                    pygame.draw.line(self.screen, (255,255,255), (self.car.p2.x, self.car.p2.y), (pt.x, pt.y), 1)

        # Render control indicators
        pygame.draw.rect(self.screen, (255,255,255), (800, 100, 40, 40), 2)  # Left
        pygame.draw.rect(self.screen, (255,255,255), (850, 100, 40, 40), 2)  # Forward/Backward
        pygame.draw.rect(self.screen, (255,255,255), (900, 100, 40, 40), 2)  # Right
        pygame.draw.rect(self.screen, (255,255,255), (850, 50, 40, 40), 2)   # Backward

        # Highlight controls based on action
        if action == 4:
            pygame.draw.rect(self.screen, (0,255,0), (850, 50, 40, 40))  # Backward
        elif action == 6:
            pygame.draw.rect(self.screen, (0,255,0), (850, 50, 40, 40))  # Backward
            pygame.draw.rect(self.screen, (0,255,0), (800, 100, 40, 40))  # Left
        elif action == 5:
            pygame.draw.rect(self.screen, (0,255,0), (850, 50, 40, 40))  # Backward
            pygame.draw.rect(self.screen, (0,255,0), (900, 100, 40, 40))  # Right
        elif action == 1:
            pygame.draw.rect(self.screen, (0,255,0), (850, 100, 40, 40))  # Forward
        elif action == 8:
            pygame.draw.rect(self.screen, (0,255,0), (850, 100, 40, 40))  # Forward
            pygame.draw.rect(self.screen, (0,255,0), (800, 100, 40, 40))  # Left
        elif action == 7:
            pygame.draw.rect(self.screen, (0,255,0), (850, 100, 40, 40))  # Forward
            pygame.draw.rect(self.screen, (0,255,0), (900, 100, 40, 40))  # Right
        elif action == 2:
            pygame.draw.rect(self.screen, (0,255,0), (800, 100, 40, 40))  # Left
        elif action == 3:
            pygame.draw.rect(self.screen, (0,255,0), (900, 100, 40, 40))  # Right

        # Render points
        text_surface = self.font.render(f'Points {self.car.points}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(0, 0))
        # Render speed
        text_surface = self.font.render(f'Speed {self.car.vel*-1}', True, pygame.Color('green'))
        self.screen.blit(text_surface, dest=(800, 0))

        # Control frame rate
        self.clock.tick(self.fps)
        # Update display
        pygame.display.update()

    # Method to close the environment
    def close(self):
        # Quit pygame
        pygame.quit()