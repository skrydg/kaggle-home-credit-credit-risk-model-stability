
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
        if column not in self.labels:
            self.labels[column] = set()
        self.labels[column].add(label)

    def get_columns_with_label(self, label):
        return [column for column, labels in self.labels if label in labels]
