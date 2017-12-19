def enhancedFeatureExtractorDigit(datum):
    
    """
    INPUT : Datum representing pixels of an image of a digit.
    OUTPUT: Returns a set of pixel features for the given datum.
    Feature 1: Whether the given pixel is a solid(1) or blank pixel (0).
    Feature 2: Count for connected blank regions (1, 2 or 3 or more regions).
    Feature 3: Count for continuous blank pixels between solid pixels.
    """
    
    
    features =  basicFeatureExtractorDigit(datum)
    # we assume we won't get a datum with 0 blank region,so features['0'] is not defined.
    
    features['1'] = 0
    features['2'] = 0
    features['>=3'] = 0

    frontier = util.Stack()
    regionNo = 0
    exploredSet = []
    
    for xPos in range(DIGIT_DATUM_WIDTH):
        for yPos in range(DIGIT_DATUM_HEIGHT):
            # check if current pixel is already visited
            if [xPos,yPos] not in exploredSet:
                #if current pixel is a blank pixel
                if (datum.getPixel(xPos,yPos) == 0):
                    frontier.push([xPos,yPos])
                    exploredSet.append([xPos,yPos])
                    regionNo += 1
                    # Running DFS
                    while not frontier.isEmpty():
                        leafNode = frontier.pop()
                        x = leafNode[0]
                        y = leafNode[1]
                        for nx,ny in findNeighbours(x,y):
                            # check if neighbour has been visited
                            if [nx,ny] not in exploredSet:
                                frontier,exploredSet = validateNeighbour(nx,ny,frontier,exploredSet,datum)
    # Adding feature values
    if regionNo == 1:
        features['1'] = 1
    elif regionNo == 2: 
        features['2'] = 1
    else:
        features['>=3'] = 1
        
    
    
    
    return features

def findNeighbours(x, y):
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def validateNeighbour(nx, ny, frontier, exploredSet, datum):
    if validPixel(nx, ny):
        # check whether pixel is a blank pixel
        if (datum.getPixel(nx, ny) == 0):
            frontier.push ([nx, ny])
            exploredSet.append([nx, ny])
    return frontier, exploredSet

def validPixel(x, y):
    return (x >= 0) and (x < DIGIT_DATUM_WIDTH) and (y >= 0) and (y < DIGIT_DATUM_HEIGHT)
    
