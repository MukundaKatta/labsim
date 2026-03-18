"""Tests for Labsim."""
from src.core import Labsim
def test_init(): assert Labsim().get_stats()["ops"] == 0
def test_op(): c = Labsim(); c.process(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Labsim(); [c.process() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Labsim(); c.process(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Labsim(); r = c.process(); assert r["service"] == "labsim"
