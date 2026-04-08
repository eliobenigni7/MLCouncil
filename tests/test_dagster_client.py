from __future__ import annotations

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_get_last_status_uses_pipeline_name_filter_and_partition_tag():
    from api.services.dagster_client import DagsterClient

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": {
                    "runsOrError": {
                        "results": [
                            {
                                "runId": "run-123",
                                "status": "SUCCESS",
                                "startTime": "1",
                                "endTime": "2",
                                "tags": [
                                    {"key": "dagster/partition", "value": "2026-04-03"},
                                    {"key": "foo", "value": "bar"},
                                ],
                            }
                        ]
                    }
                }
            }

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            self.calls.append((url, json))
            DummyAsyncClient.last_call = (url, json)
            return DummyResponse()

    with patch("api.services.dagster_client.httpx.AsyncClient", DummyAsyncClient):
        client = DagsterClient("http://dagster:3000/graphql")
        status = await client.get_last_status()

    url, payload = DummyAsyncClient.last_call
    assert url == "http://dagster:3000/graphql"
    assert payload["variables"]["jobName"] == "daily_pipeline"
    assert "pipelineName" in payload["query"]
    assert "partition" not in payload["query"]
    assert status.last_run_id == "run-123"
    assert status.partition == "2026-04-03"


@pytest.mark.asyncio
async def test_trigger_run_uses_partition_backfill_for_partitioned_job():
    from api.services.dagster_client import DagsterClient

    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            if "GetPartitionKeys" in json["query"]:
                return DummyResponse(
                    {
                        "data": {
                            "partitionSetOrError": {
                                "partitionsOrError": {
                                    "results": [{"name": "2026-04-02"}, {"name": "2026-04-03"}]
                                }
                            }
                        }
                    }
                )
            DummyAsyncClient.last_call = (url, json)
            return DummyResponse(
                {
                    "data": {
                        "launchPartitionBackfill": {
                            "__typename": "LaunchBackfillSuccess",
                            "backfillId": "bf-123",
                            "launchedRunIds": ["run-456"],
                        }
                    }
                }
            )

    with patch("api.services.dagster_client.httpx.AsyncClient", DummyAsyncClient):
        client = DagsterClient("http://dagster:3000/graphql")
        run_id = await client.trigger_run("2026-04-03")

    _, payload = DummyAsyncClient.last_call
    assert "launchPartitionBackfill" in payload["query"]
    assert payload["variables"]["backfillParams"]["partitionNames"] == ["2026-04-03"]
    assert payload["variables"]["backfillParams"]["selector"]["partitionSetName"] == "daily_pipeline_partition_set"
    assert payload["variables"]["backfillParams"]["runConfigData"] == {}
    assert payload["variables"]["backfillParams"]["tags"] == [
        {"key": "dagster/partition", "value": "2026-04-03"}
    ]
    assert run_id == "run-456"


@pytest.mark.asyncio
async def test_trigger_run_rejects_unavailable_partition_with_latest_hint():
    from api.services.dagster_client import DagsterClient, DagsterClientError

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "data": {
                    "partitionSetOrError": {
                        "partitionsOrError": {
                            "results": [{"name": "2026-04-01"}, {"name": "2026-04-02"}]
                        }
                    }
                }
            }

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return DummyResponse()

    with patch("api.services.dagster_client.httpx.AsyncClient", DummyAsyncClient):
        client = DagsterClient("http://dagster:3000/graphql")
        with pytest.raises(DagsterClientError, match="Latest valid partition: 2026-04-02"):
            await client.trigger_run("2026-04-03")
