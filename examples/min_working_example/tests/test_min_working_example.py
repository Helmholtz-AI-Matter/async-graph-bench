import os
import subprocess
import sys

import pytest

from examples.min_working_example.run import (
    AdditionCalculator,
    NoiseAdder,
    NumberTupleDataSource,
    SquareCalculator,
)


@pytest.mark.asyncio
async def test_data_source_and_calculators():
    source = NumberTupleDataSource()
    assert len(source) == 4
    assert list(source.iter_ids()) == [0, 1, 2, 3]

    assert SquareCalculator()({"second_number": [2, 3]}) == {
        "second_number_squared": [4, 9]
    }
    assert await AdditionCalculator()(
        {"first_number_noisy": [1.5, 2.5], "second_number_squared": [4, 9]}
    ) == {"sum": [5.5, 11.5]}


@pytest.mark.asyncio
async def test_noise_adder_with_mock_resource():
    class MockNoiseResource:
        def generate_noise(self, length):
            return [0.0] * length

    result = await NoiseAdder()({"first_number": [1.0, 2.0]}, MockNoiseResource())

    assert result == {"first_number_noisy": [1.0, 2.0]}


@pytest.mark.slow
def test_example_runs_with_real_noise_resource(tmp_path):
    environment = os.environ.copy()
    environment["ASYNC_GRAPH_EXAMPLE_ITERATIONS"] = "1"
    environment["ASYNC_GRAPH_EXAMPLE_DATA_PATH"] = str(tmp_path / "data")

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "run.py"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "data").exists()
