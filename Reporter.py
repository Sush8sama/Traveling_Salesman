import math
import statistics
import sys
import time

import numpy as np

import r0885724
import pandas as pd
import matplotlib.pyplot as plt



# Class to report basic results of an evolutionary algorithm
class Reporter:

    def __init__(self, filename):
        self.allowedTime = 300
        self.numIterations = 0
        self.filename = filename + ".csv"
        self.delimiter = ','
        self.startTime = time.time()
        self.writingTime = 0
        outFile = open(self.filename, "w")
        # outFile.write("# Student number: " + filename + "\n")
        outFile.write("Iteration, Elapsed time, Mean value, Best value, Cycle\n")
        outFile.close()

    # Append the reported mean objective value, best objective value, and the best tour
    # to the reporting file.
    #
    # Returns the time that is left in seconds as a floating-point number.
    def report(self, meanObjective, bestObjective, bestSolution):
        if (time.time() - self.startTime < self.allowedTime + self.writingTime):
            start = time.time()

            outFile = open(self.filename, "a")
            outFile.write(str(self.numIterations) + self.delimiter)
            outFile.write(str(start - self.startTime - self.writingTime) + self.delimiter)
            outFile.write(str(meanObjective) + self.delimiter)
            outFile.write(str(bestObjective) + self.delimiter)
            for i in range(bestSolution.size):
                outFile.write(str(bestSolution[i]) + ";")
            outFile.write('\n')
            outFile.close()

            self.numIterations += 1
            self.writingTime += time.time() - start
        return (self.allowedTime + self.writingTime) - (time.time() - self.startTime)

if __name__ == '__main__':
    tour = [1000, 500, 1000]
    # filename = r"C:\Users\wangq\kul\Traveling_Salesman\Tours\tour"
    best_cost_1 = [83557.18694408631,80198.1226303031,83093.85486371486,81597.20943248857, 83819.71640650899,80891.49383821996,
                 81601.87237479247, 81367.80916334814, 82514.4231432483,82948.60197522023, 81857.0215430049,81114.41231813145,
                 81755.98217836284,81660.86794336647,79972.91466735827,82120.19716685828,82586.86352943731,81929.00260155645,
                83116.86498853561,81065.13785296842,81317.86314283787,81030.662298179,82068.78470227227, 82174.58171914233,
                 83652.53359367524,80409.10440200994,81854.49833803913,81748.55467767807,82104.07508464801, 81856.9329149985]
    best_cost_2 = [154592.74878199745,158748.9190403178,157190.47820993763,157469.2944234075,155513.00959636676,153742.2397453586,
                   153669.18279104904,153021.80476924387, 151583.53478264276,153970.1982848241,160342.3354556535,154108.56692318464
                   ,153395.8833328779, 159472.01834281904,157490.58388387074,156461.66449135193,155325.72827903976,156061.25955279177,
                   150351.0110181373,154031.08860670586, 152930.59474395332, 154343.72302683865, 156223.5615377586,151951.59069531545,
                   155741.90103660978,153806.4620597069, 156377.3060482127,156164.5451892212,155633.0024922406,159541.19207783716]
    best_cost_3 = [200050.64793557284,204901.3296085998,203660.8277352792,205743.4457471778,205216.56985130798,206853.340527285,
                   199366.24672754723,205655.0186154005,203662.3413380738, 202018.64493282404,204790.40955647966, 204067.86533186422,
                   204960.24563459487,203603.42143442342,205589.24500492666,201249.34755983073, 206182.20454655326,203261.68818482355,
                   203511.72743554518,205119.7549846033,204766.43478632692, 199355.06844675622,199760.25364627078,201733.71859153445,
                   202651.53461679173,204504.7530795952,202685.89572494035,201999.3966081689,206772.7590737943,200562.2134342855]
    best_cost = []
    mean_cost = []
    best_costs = 10000000
    best_solution = None

    for i in range(1):
        filename = r"C:\Users\wangq\kul\Traveling_Salesman\Tours\tour" + str(tour[i]) + ".csv"
        
      
        a = r0885724.r0885724()
        answer = a.optimize(filename)
        best_cost.append(answer[0])
        mean_cost.append(answer[1])
        solution = answer[2]
        if best_cost[i] < best_costs:
            best_costs = best_cost[i]
            best_solution = solution

        data = pd.read_csv('r0885724.csv')
        print(data.columns)
        time = data[' Elapsed time']
        mean_value = data[' Mean value']
        best_value = data[' Best value']
        plt.figure(figsize=(10, 6))
        
        # Plot the Mean value
        plt.plot(time, mean_value, label='Mean Value', color='blue', linewidth=2)
        
        # Plot the Best value
        plt.plot(time, best_value, label='Best Value', color='red', linewidth=2)
        
        # Add labels and title
        plt.xlabel('Elapsed time')
        plt.ylabel('Value')
        plt.title('Mean and Best Value Over Time')
        
        # Add a legend
        plt.legend()
        
        # Display the plot
        plt.show()
        print(i)
    # print(12455)
    # bins = np.arange(199000, 207000, 500 )  

    # # Plot histograms for both arrays
    # plt.figure(figsize=(12, 6))

    # # Plot histogram for array1
    # plt.hist(best_cost_3, bins=bins, alpha=0.5, label='Best costs', edgecolor='black')

    # # Plot histogram for array2
    # # plt.hist(mean_cost, bins=bins, alpha=0.5, label='Mean costs', edgecolor='black')

    # # Labels and title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histograms of Best Costs for tour 1000')
    # print("Best cost:", statistics.mean(best_cost_3), statistics.pstdev(best_cost_3))
    # # print("Mean cost:", statistics.mean(mean_cost), statistics.pstdev(mean_cost))
    # # print("best Solution: " ,best_solution)
    # plt.show()

    

