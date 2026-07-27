import math

from pure_pursuit.utils import pt_to_pt_distance, sgn

"""
credit to purdue sigbots for pure pursuit function
"""

# limo max speed 0.5 m/s
max_speed = 0.5

class PurePursuit:
    def __init__(self, lookAheadDis):
        self.last_found_index = 0
        self.lookAheadDis = lookAheadDis

    def compute_control(self, path, currentPos, currentHeading):
        # extract currentX and currentY
        currentX = currentPos[0]
        currentY = currentPos[1]

        # use for loop to search intersections
        lastFoundIndex = self.last_found_index
        intersectFound = False
        startingIndex = lastFoundIndex

        for i in range (startingIndex, len(path)-1):

            # beginning of line-circle intersection code
            x1 = path[i][0] - currentX
            y1 = path[i][1] - currentY
            x2 = path[i+1][0] - currentX
            y2 = path[i+1][1] - currentY
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt (dx**2 + dy**2)
            D = x1*y2 - x2*y1
            discriminant = (self.lookAheadDis**2) * (dr**2) - D**2

            if discriminant >= 0:
                sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
                sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

                sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
                sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
                # end of line-circle intersection code

                minX = min(path[i][0], path[i+1][0])
                minY = min(path[i][1], path[i+1][1])
                maxX = max(path[i][0], path[i+1][0])
                maxY = max(path[i][1], path[i+1][1])

                # if one or both of the solutions are in range
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

                    foundIntersection = True

                    # if both solutions are in range, check which one is better
                    if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
                        # make the decision by compare the distance between the intersections and the next point in path
                        if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2
                    
                    # if not both solutions are in range, take the one that's in range
                    else:
                        # if solution pt1 is in range, set that as goal point
                        if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2
                    
                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
                        # update lastFoundIndex and exit
                        lastFoundIndex = i
                        break
                    else:
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        lastFoundIndex = i+1
                    
                # if no solutions are in range
                else:
                    foundIntersection = False
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

        # obtained goal point, now compute turn vel
        # initialize proportional controller constant
        Kp = 3

        # calculate absTargetAngle with the atan2 function
        absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/math.pi
        if absTargetAngle < 0: absTargetAngle += 360

        # compute turn error by finding the minimum angle
        turnError = absTargetAngle - currentHeading
        if turnError > 180 or turnError < -180 :
            turnError = -1 * sgn(turnError) * (360 - abs(turnError))
        
        # apply proportional controller
        turnVel = Kp*turnError

        # update last found index variable
        self.last_found_index = lastFoundIndex

        # calculate speed - slow down in sharp turns
        # TODO: IF DOESNT WORK SET TO A CONSTANT
        linear_speed = max_speed / (1 + turnVel)

        return turnVel, linear_speed
