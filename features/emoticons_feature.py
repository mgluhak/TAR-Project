from features.feature import Feature
import dataset.dataset_reader as dr


class EmoticonFeature(Feature):
    @staticmethod
    def is_emoticon(token):
        if len(token) < 2:
            return False
        return True if token[0] in [':', ';', '='] else False

    def is_nose_emoticon(self, token):
        if len(token) < 3:
            return False
        return True if (self.is_emoticon(token) and token[1] is '-') else False

    @staticmethod
    def is_reverse_emoticon(token):
        if len(token) < 2:
            return False
        return True if token[-1] in [':', ';', '='] else False

    def is_happy_emoticon(self, token):
        if len(token) < 2:
            return False
        return True if (self.is_emoticon(token) and token[-1] in [')', ']', 'P', 'D']) else False

    def extract_feature(self, user_id, user_tweets):
        no_of_reverse = 0
        no_of_noses = 0
        no_of_happy = 0
        no_of_emoticons = 0
        no_of_tokens = 0
        for tweet in user_tweets:
            for token in tweet:
                no_of_reverse += 1 if self.is_reverse_emoticon(token) else 0
                no_of_noses += 1 if self.is_nose_emoticon(token) else 0
                no_of_happy += 1 if self.is_happy_emoticon(token) else 0
                no_of_emoticons += 1 if self.is_emoticon(token) else 0
                no_of_tokens += 1
        return no_of_noses, no_of_happy, no_of_reverse, no_of_emoticons, no_of_tokens


data = dr.load_dataset()
emoticon = EmoticonFeature()
print(emoticon.extract_feature('36b2593435e1bed13eb138c1973c13ed', data['36b2593435e1bed13eb138c1973c13ed'].tweets))
