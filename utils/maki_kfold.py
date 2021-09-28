from sklearn.utils import shuffle
import copy


class Unit:
    __slots__ = ('label', 'path', 'index')

    def __init__(self, label, path, index):
        self.label = label
        self.path = path
        self.index = index


class MakiKFoldBalance:

    def __init__(self, n_groups: int, *, shuffle_data=False):
        """
        Split dataset into `n_groups` and provide test/train data

        Parameters
        ----------
        n_groups : int
            Number of groups
        balance_all_batches : bool
            Keep balance of each group or not
        shuffle_data : bool
            Shuffle data after create group

        """
        self._n_groups = n_groups
        self._shuffle_data = shuffle_data

    def split(self, data_path: list, labels: list, num_classes: int):
        """
        
        Returns
        -------
        train_list : list
        test_list : list

        """
        # Info about each class
        # class_1: [indx_1, indx_2, ...],
        # class_n: [indx_1, indx_2, ...]
        data_collector = {}
        for i in range(len(labels)):
            if data_collector.get(str(labels[i])) is None:
                data_collector[str(labels[i])] = [Unit(labels[i], data_path[i], i)]
            else:
                data_collector[str(labels[i])] += [Unit(labels[i], data_path[i], i)]

        for key in data_collector:
            data_collector[key] = shuffle(data_collector[key])

        counter = 0
        single_batch = len(labels) // self._n_groups
        single_class_num_on_test = single_batch // num_classes
        num_of_single_class = len(labels) // num_classes

        while counter != self._n_groups:
            test_list = []
            train_list = []

            for key in data_collector:
                for i in range(counter * single_class_num_on_test, (counter + 1) * single_class_num_on_test):
                    test_list.append(data_collector[key][i].index)

                for i in range(0, counter * single_class_num_on_test):
                    train_list.append(data_collector[key][i].index)

                for i in range((counter + 1) * single_class_num_on_test, num_of_single_class):
                    train_list.append(data_collector[key][i].index)

            if self._shuffle_data:
                train_list = shuffle(train_list)
                test_list = shuffle(test_list)

            yield train_list, test_list
            counter += 1


