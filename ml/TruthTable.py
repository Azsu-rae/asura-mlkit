
class TruthTable:

    def __init__(self, columnNames, outputValues):
        if pow(2, len(columnNames)) != len(outputValues):
            raise ValueError("Incompatible columns and outputValues!")

        self.outputValues = outputValues
        columnNames.append("Output")
        self.columnNames = columnNames

    @staticmethod
    def lengthen(bools, length):
        while len(bools) < length:
            bools.extend(list(bools))

    @staticmethod
    def baseBools(bits):
        baseBools, length = [], pow(2, bits)
        block = length
        for i in range(bits):
            bools = [j >= block//2 for j in range(block)]
            TruthTable.lengthen(bools, length)
            baseBools.append(bools)
            block //= 2

        result = []
        for i in range(pow(2, bits)):
            result.append([row[i] for row in baseBools])
 
        return result

    def values(self):
        values = TruthTable.baseBools(len(self.columnNames)-1)
        for i, row in enumerate(values):
            row.append(self.outputValues[i])

        return values

    def drawTable(self):
        s, length, columnLimitations = ["|"], 1, [0]
        for columnName in self.columnNames:

            length += len(columnName)+1
            s.append(" " + columnName)
 
            length += 2
            s.append(" |")
            columnLimitations.append(length-1)

        s.append("\n")
        for i in range(length):
            if i in columnLimitations:
                s.append("+")
            else:
                s.append("-")

        s.append("\n")
        baseBools = self.values()
        for row in baseBools:
            first = True
            for col, baseBool in enumerate(row):
                nameLength = len(self.columnNames[col])
                halfway = nameLength//2
                if first:
                    s.append("|")
                    first = False
                s.append(" " * (halfway+1))
                s.append("1" if baseBool else "0")
                s.append(" " * (nameLength - halfway))
                s.append("|")
            s.append("\n")

        print("".join(s))

