"""Stats: the base stats every character carries."""
import pytest

from carriage_return.entity import Entity
from carriage_return.inventory import Inventory
from carriage_return.stats import STATS, Stats


class Rock:
    """A stand-in item with only the weight Inventory looks at."""
    def __init__(self, weight):
        self.weight = weight
        self.length = 10


def test_defaults():
    stats = Stats(Entity('player'))
    assert stats.hit_points == 10
    assert stats.magic_cooldown == 0
    assert stats.magic_threshold == 10
    assert len(stats.as_dict()) == len(STATS)


def test_abbreviation_access_matches_attributes():
    stats = Stats(Entity('player'))
    stats['ST'] = 14
    assert stats.strength == 14
    assert stats['SP'] == stats.stamina


def test_overrides_and_unknown_stat():
    stats = Stats(Entity('player'), dexterity=3)
    assert stats.dexterity == 3
    with pytest.raises(TypeError):
        Stats(Entity('player'), luck=7)


def test_stats_is_a_component_of_its_owner():
    owner = Entity('player')
    owner.stats = Stats(owner)
    assert owner.has_component('stats')
    assert owner.stats.parent_entity is owner


def test_inventory_weight_limit_is_owner_carry_capacity():
    owner = Entity('player')
    owner.stats = Stats(owner, carry_capacity=5)
    inv = Inventory(owner, allowed_slots=['hand'])
    assert inv.max_weight == 5
    assert inv.check_entity_add(Rock(4), 'hand', actor=owner)[0]
    allowed, reasons = inv.check_entity_add(Rock(6), 'hand', actor=owner)
    assert not allowed and "entity is too heavy" in reasons


def test_inventory_without_stats_has_no_weight_limit():
    inv = Inventory(Entity('maze'))
    assert inv.max_weight is None
    assert inv.check_entity_add(Rock(1000), 'anywhere', actor=None)[0]
