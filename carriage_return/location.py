from carriage_return.events import EventEmitter, Event

from carriage_return.entity import Component


class Location(Component):
    """Stores the location of an entity: what container is it in, and which slot within the container
    """

    def __init__(self, entity, container, slot):
        Component.__init__(self, entity, component_type='location')

        # Container and slot are held as one tuple and replaced in a single
        # assignment, never edited apart. A location only means anything as a
        # pair -- a slot is an index into a particular container -- so writing
        # the two separately let another thread observe a new container with
        # the previous container's slot, and index it out of bounds. Read the
        # pair through `place` wherever both halves are needed at once.
        self._place = (None, None)

        # emitted when this actual location changes
        self.changed = EventEmitter(source=self, type='location_change', event_class=LocationChangeEvent)

        # emitted when this location or any parent changes
        self.global_changed = EventEmitter(source=self, type='global_location_change', event_class=LocationChangeEvent)

        self.update(container, slot)

    @property
    def place(self):
        """This location as one ``(container, slot)`` tuple.

        Read this, rather than ``container`` and ``slot`` separately, whenever
        both halves are used together -- indexing the container by the slot,
        most of all. The tuple is never mutated, so the pair it returns always
        belongs together even if another thread is moving this entity. The
        individual properties are fine on their own; it is only the
        combination that must not be read in two steps.
        """
        return self._place

    @property
    def container(self):
        return self._place[0]

    @property
    def slot(self):
        return self._place[1]

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
        old_container, old_slot = self._place

        if container is not None:
            slot = container.inventory.slot_type(slot)
        if container is old_container and slot == old_slot:
            return

        # reconnect events so that we get global change notifications
        if container is not old_container:
            if old_container is not None:
                old_container.location.global_changed.disconnect(self._container_moved)
            if container is not None:
                container.location.global_changed.connect(self._container_moved)

        # publish the new location in one write; see the note in __init__
        self._place = (container, slot)

        # update inventories
        if old_container is not None:
            old_container.inventory._remove_entity(entity=self.parent_entity, slot=old_slot)
        if container is not None:
            container.inventory._add_entity(entity=self.parent_entity, slot=slot)

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