from enum import Enum


class Gender(Enum):
    MALE = 0
    FEMALE = 1


class AgeGroup(Enum):
    _18_24 = 0
    _25_34 = 1
    _35_49 = 2
    _50_64 = 3
    _65_xx = 4


class TweetMapEntry:
    def __init__(self, gender, age_group, tweets):
        self.gender = gender
        self.age_group = age_group
        self.tweets = tweets

    def get_gender(self):
        return self.gender

    def get_age_group(self):
        return self.age_group

    def get_tweets(self):
        return self.tweets
