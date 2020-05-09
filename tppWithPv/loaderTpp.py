from loader import Loader


class LoaderTpp(Loader):

    def iterateTrainId(self):
        for trainerId in [1,2,3]:
            yield trainerId
