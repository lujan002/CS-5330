Your name and the names of any group members:
Luke Jansen

Links/URLs to any videos you created and want to submit as part of your report:
n/a

What operating system and IDE you used to run and compile your code:
Cursor, Ubuntu 22.04

Instructions for running your executables:
1. Navigate to the project2/build directory
2. Run this command to generate the csv file of feature vectors for the entire dataset
./dirToFeatureVec <datasetPath> <featureSet> <featureVectorCSVPath>
For example: 
./dirToFeatureVec ../olympus baseline ../featureVectors1.csv
3. Run this command to perform image retreival 
./imgMatch <imagePath> <featureVectorCSVPath> <distanceMetric> <numMatches>
For example:
./imgMatch ../olympus/pic.1016.jpg ../featureVectors1.csv least_squares 10

<featureSet> arguments: 
baseline
color
spatial_color
sobel
depth
gabor

<distanceMatric> arguements:
least_squares
single_hist_intersect
chi_squared_distance
cosine_distance

Instructions for testing any extensions you completed:
Follow above steps using "gabor" as the <featureSet> arguement

Whether you are using any time travel days, and how many:
No time travel days used

Any other information about your code you wish to include:
n/a
