
def desc(title, values):
    print(f"\n{title}\n")
    first = True
    for i in range(len(values)):
        if i < 10 or i > len(values)-10:
            print(f"student {i}: {values[i]}")
        elif first:
            print(".")
            print(".")
            print(".")
            print(".")
            first = False

import numpy as np

np.random.seed(101)
grades_of_students = np.random.randint(0, 21, size=(100, 5))

modules = ["THL", "DSS", "DB", "RE", "SE"]

desc("Mean", np.mean(grades_of_students, axis=1))
desc("Variance", np.var(grades_of_students, axis=1))
desc("Median", np.median(grades_of_students, axis=1))
desc("Standard Deviation", np.std(grades_of_students, axis=1))
