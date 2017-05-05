from abc import ABC, abstractmethod

class Feature(ABC):

    def __init__(self):
        self.map = {}

    @abstractmethod
    def extract_feature(self, user, tweets):
        pass

    def retrieve(self, user, tweets=None):
        if (user in self.map) and tweets is not None:
            self.extract_feature(user, tweets)
            return self.map[user]
        elif user in self.map:
            return self.map[user]
        else:
            raise ValueError("User with given id does not exist!")
