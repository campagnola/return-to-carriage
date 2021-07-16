from vispy.util.event import EventEmitter, Event


class Location:
    """Stores the location of an entity: what container is it in, and which slot within the container
    """

    def __init__(self, entity, container, slot):
        self.entity = entity
        self.container = None
        self.slot = None

        # emitted when this actual location changes
        self.changed = EventEmitter(source=self, type='location_change', event_class=LocationChangeEvent)

        # emitted when this location or any parent changes
        self.global_changed = EventEmitter(source=self, type='global_location_change', event_class=LocationChangeEvent)

        self.update(container, slot)

    def set_slot(self, slot):
        self.update(self.container, slot)

    def update(self, container, slot):
        """Change the location referred to by this object.

        This is the _one_ correct place from which entity locations should be changed.

        Parameters
        ==========
        container : Entity
            Update the entity this location points to.
        slot : object
            Update the slot (within container's inventory) this location points to
        """
        old_container = self.container
        old_slot = self.slot
        changed = False
        
        if container is not old_container:
            self.container = container
            changed = True
            # reconnect events so that we get global change notifications
            if old_container is not None:
                old_container.location.global_changed.disconnect(self._container_moved)
            if container is not None:
                container.location.global_changed.connect(self._container_moved)

        if self.container is not None:
            slot = self.container.inventory.slot_type(slot)
        if slot != old_slot:
            self.slot = slot
            changed = True

        if changed:
            # update inventories
            if old_container is not None:
                old_container.inventory._remove_entity(entity=self.entity, slot=old_slot)
            if self.container is not None:
                self.container.inventory._add_entity(entity=self.entity, slot=self.slot)
            
            # emit events
            old_loc = (old_container, old_slot)
            self.changed(old_location=old_loc)
            self.global_changed(moved_entity=self, old_location=old_loc)

    def _container_moved(self, event):
        self.global_changed(moved_entity=event.moved_entity, old_location=event.old_location)

    @property
    def global_location(self):
        """The global (maze) location that contains this location
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


class LocationChangeEvent(Event):
    pass