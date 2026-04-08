from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from runtime_env import load_runtime_env

load_runtime_env()

DAGSTER_GRAPHQL_URL = os.getenv(
    "DAGSTER_GRAPHQL_URL", "http://localhost:3000/graphql"
)
DAGSTER_JOB_NAME = "daily_pipeline"
DAGSTER_REPOSITORY_NAME = os.getenv("DAGSTER_REPOSITORY_NAME", "__repository__")
DAGSTER_REPOSITORY_LOCATION_NAME = os.getenv(
    "DAGSTER_REPOSITORY_LOCATION_NAME", "pipeline.py"
)
DAGSTER_PARTITION_SET_NAME = os.getenv(
    "DAGSTER_PARTITION_SET_NAME", f"{DAGSTER_JOB_NAME}_partition_set"
)


@dataclass
class PipelineStatus:
    last_run_id: str | None
    status: str
    start_time: str | None
    end_time: str | None
    partition: str | None


_LAUNCH_RUN_MUTATION = """
mutation LaunchRun(
  $jobName: String!,
  $repositoryName: String!,
  $repositoryLocationName: String!,
  $runConfigData: RunConfigData,
  $executionMetadata: ExecutionMetadata
) {
  launchRun(executionParams: {
    selector: {
      repositoryName: $repositoryName,
      repositoryLocationName: $repositoryLocationName,
      jobName: $jobName,
    },
    runConfigData: $runConfigData,
    executionMetadata: $executionMetadata,
  }) {
    __typename
    ... on LaunchRunSuccess {
      run { runId status }
    }
  }
}
"""

_LAUNCH_PARTITION_BACKFILL_MUTATION = """
mutation LaunchPartitionBackfill($backfillParams: LaunchBackfillParams!) {
  launchPartitionBackfill(backfillParams: $backfillParams) {
    __typename
    ... on LaunchBackfillSuccess {
      backfillId
      launchedRunIds
    }
  }
}
"""

_RUNS_QUERY = """
query GetRuns($jobName: String!, $limit: Int!) {
  runsOrError(filter: { pipelineName: $jobName }, limit: $limit) {
    ... on Runs {
      results {
        runId
        status
        startTime
        endTime
        tags {
          key
          value
        }
      }
    }
  }
}
"""

_PARTITION_KEYS_QUERY = """
query GetPartitionKeys(
  $repositoryName: String!,
  $repositoryLocationName: String!,
  $partitionSetName: String!
) {
  partitionSetOrError(
    repositorySelector: {
      repositoryName: $repositoryName,
      repositoryLocationName: $repositoryLocationName,
    },
    partitionSetName: $partitionSetName,
  ) {
    __typename
    ... on PartitionSet {
      partitionsOrError {
        __typename
        ... on Partitions {
          results { name }
        }
      }
    }
  }
}
"""


class DagsterClientError(Exception):
    pass


class DagsterClient:
    def __init__(self, graphql_url: str = DAGSTER_GRAPHQL_URL):
        self._url = graphql_url

    @staticmethod
    def _format_timestamp(value: float | int | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, (float, int)):
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        return str(value)

    async def trigger_run(self, partition: str | None = None) -> str:
        if partition:
            latest_partition = await self.get_latest_partition_key()
            partition_keys = await self.get_partition_keys()
            if partition not in partition_keys:
                latest_msg = f" Latest valid partition: {latest_partition}." if latest_partition else ""
                raise DagsterClientError(
                    f"Partition {partition} is not available in Dagster.{latest_msg}"
                )

        query = _LAUNCH_RUN_MUTATION
        variables = {
            "jobName": DAGSTER_JOB_NAME,
            "repositoryName": DAGSTER_REPOSITORY_NAME,
            "repositoryLocationName": DAGSTER_REPOSITORY_LOCATION_NAME,
            "runConfigData": {},
            "executionMetadata": None,
        }

        if partition:
            query = _LAUNCH_PARTITION_BACKFILL_MUTATION
            variables = {
                "backfillParams": {
                    "selector": {
                        "partitionSetName": DAGSTER_PARTITION_SET_NAME,
                        "repositorySelector": {
                            "repositoryName": DAGSTER_REPOSITORY_NAME,
                            "repositoryLocationName": DAGSTER_REPOSITORY_LOCATION_NAME,
                        },
                    },
                    "partitionNames": [partition],
                    "forceSynchronousSubmission": True,
                    "tags": [{"key": "dagster/partition", "value": partition}],
                    "runConfigData": {},
                }
            }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._url,
                json={
                    "query": query,
                    "variables": variables,
                },
            )

        if resp.status_code != 200:
            raise DagsterClientError(f"Dagster GraphQL error {resp.status_code}: {resp.text}")

        data = resp.json()
        errors = data.get("errors")
        if errors:
            raise DagsterClientError(f"Dagster errors: {errors}")

        if partition:
            launch_backfill = data["data"]["launchPartitionBackfill"]
            if launch_backfill.get("__typename") != "LaunchBackfillSuccess":
                raise DagsterClientError(f"Dagster launch failed: {launch_backfill}")
            run_ids = launch_backfill.get("launchedRunIds") or []
            if not run_ids:
                raise DagsterClientError(f"No run id returned from Dagster backfill: {launch_backfill}")
            return run_ids[0]

        launch_run = data["data"]["launchRun"]
        if launch_run.get("__typename") != "LaunchRunSuccess":
            raise DagsterClientError(f"Dagster launch failed: {launch_run}")

        return launch_run["run"]["runId"]

    async def get_partition_keys(self) -> list[str]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                self._url,
                json={
                    "query": _PARTITION_KEYS_QUERY,
                    "variables": {
                        "repositoryName": DAGSTER_REPOSITORY_NAME,
                        "repositoryLocationName": DAGSTER_REPOSITORY_LOCATION_NAME,
                        "partitionSetName": DAGSTER_PARTITION_SET_NAME,
                    },
                },
            )

        if resp.status_code != 200:
            raise DagsterClientError(f"Dagster GraphQL error {resp.status_code}: {resp.text}")

        data = resp.json()
        errors = data.get("errors")
        if errors:
            raise DagsterClientError(f"Dagster errors: {errors}")

        partition_set = data.get("data", {}).get("partitionSetOrError", {})
        partitions = partition_set.get("partitionsOrError", {}).get("results", [])
        return [item["name"] for item in partitions if item.get("name")]

    async def get_latest_partition_key(self) -> str | None:
        keys = await self.get_partition_keys()
        return keys[-1] if keys else None

    async def get_last_status(self) -> PipelineStatus:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                self._url,
                json={
                    "query": _RUNS_QUERY,
                    "variables": {"jobName": DAGSTER_JOB_NAME, "limit": 1},
                },
            )

        if resp.status_code != 200:
            return PipelineStatus(
                last_run_id=None, status="unreachable",
                start_time=None, end_time=None, partition=None,
            )

        data = resp.json()
        errors = data.get("errors")
        if errors:
            return PipelineStatus(
                last_run_id=None, status="error",
                start_time=None, end_time=None, partition=None,
            )

        results = data.get("data", {}).get("runsOrError", {}).get("results", [])
        if not results:
            return PipelineStatus(
                last_run_id=None, status="no_runs",
                start_time=None, end_time=None, partition=None,
            )

        last = results[0]
        partition = None
        for tag in last.get("tags", []):
            if tag.get("key") == "dagster/partition":
                partition = tag.get("value")
                break
        return PipelineStatus(
            last_run_id=last["runId"],
            status=last["status"],
            start_time=self._format_timestamp(last.get("startTime")),
            end_time=self._format_timestamp(last.get("endTime")),
            partition=partition,
        )
