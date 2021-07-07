

class Location:
    """Stores the location of an entity: what container is it in, and which slot within the container
    """
    def __init__(self, container, slot):
        self.container = container
        self.slot = slot

    def update(self, container, slot):
        self.container = container
        self.slot = slot

    @property
    def maze_location(self):
        """The maze location that contains this location
        """
        location = self
        while True:
            if location.container is None:
                return None
            if location.container.type.isa('maze'):
                return location
            else:
                location = location.container.location

    def __iter__(self):
        yield self.container
        yield self.slot

    def __repr__(self):
        return f"<Location {repr(self.container)}, {repr(self.slot)}>"
