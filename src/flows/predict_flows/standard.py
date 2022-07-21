from src.classes.SessionManager import SessionManager

from src.functions.printing_and_logging import print_end


class ModuleClass(SessionManager):
    def __init__(self, args):
        SessionManager.__init__(self, args)

    def run(self):
        """
        Orchestrator for this class. Here you should specify all the actions you want this class to perform.
        """
        self.prepare()
        # todo: load model
        # todo: type of model
        # todo: load features
        # todo: function to calculate predictors based on raw features + treat missing
        # todo: predict
        # todo: save or push predictions
        print_end()