class ActionError(Exception):
    """Raised when a requested player action fails.
    """
    def __init__(self, reason):
        self.reason = reason
        Exception.__init__(reason)
    