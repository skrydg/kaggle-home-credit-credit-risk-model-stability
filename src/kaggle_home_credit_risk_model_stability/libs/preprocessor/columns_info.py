
class ColumnsInfo:
    def __init__(self):
        self.labels = {}
    
    def get_labels(self, column):
        if column not in self.labels:
            self.labels[column] = set()
        return self.labels[column]
    
    def delete_label(self, column, label):
        assert(column in self.labels)
        self.labels[column].delete(label)
        
    def add_label(self, column, label):
        self.add_labels(column, set([label]))

    def add_labels(self, column, labels):
        if column not in self.labels:
            self.labels[column] = set()
        self.labels[column] = self.labels[column] | labels

    def get_columns_with_label(self, label):
        return [column for column, labels in self.labels.items() if label in labels]
