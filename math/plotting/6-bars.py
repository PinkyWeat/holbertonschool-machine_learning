#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Create a stacked bar graph
plt.bar(np.arange(3), fruit[0], width=0.5, color=colors[0], label='apples')
for i in range(1, 4):
    plt.bar(np.arange(3), fruit[i], bottom=np.sum(fruit[:i], axis=0), width=0.5, color=colors[i], label=['bananas', 'oranges', 'peaches'][i-1])

plt.xticks(np.arange(3), ['Farrah', 'Fred', 'Felicia'])
plt.ylabel('Quantity of Fruit')
plt.ylim([0, 80])
plt.yticks(np.arange(0, 81, 10))

plt.legend()
plt.title('Number of Fruit per Person')

# Show the plot
plt.show()
