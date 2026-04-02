from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

DAGSTER_GRAPHQL_URL = os.getenv(
    "DAGSTER_GRAPHQL_URL", "http://localhost:3000/graphql"
)
DAGSTER_JOB_NAME = "daily_pipeline"


@dataclass
class PipelineStatus:
    last_run_id: str | None
    status: str
    start_time: str | None
    end_time: str | None
    partition: str | None


_LAUNCH_RUN_MUTATION = """
mutation LaunchRun($jobName: String!, $runConfigData: RunConfigData) {
  launchRun(executionParams: {
    jobName: $jobName,
    runConfigData: $runConfigData,
  }) {
    run { runId status }
  }
}
"""

_RUNS_QUERY = """
query GetRuns($jobName: String!, $limit: Int!) {
  runsOrError(filter: { jobName: $jobName }, limit: $limit) {
    ... on Runs {
      results {
        runId
        status
        startTime
        endTime
        partition
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

    async def trigger_run(self, partition: str | None = None) -> str:
        run_config = {}
        if partition:
            run_config["ops"] = {"context": {"config": {"partition": partition}}}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._url,
                json={
                    "query": _LAUNCH_RUN_MUTATION,
                    "variables": {"jobName": DAGSTER_JOB_NAME, "runConfigData": run_config},
                },
            )

        if resp.status_code != 200:
            raise DagsterClientError(f"Dagster GraphQL error {resp.status_code}: {resp.text}")

        data = resp.json()
        errors = data.get("errors")
        if errors:
            raise DagsterClientError(f"Dagster errors: {errors}")

        run_id = data["data"]["launchRun"]["run"]["runId"]
        return run_id

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
        return PipelineStatus(
            last_run_id=last["runId"],
            status=last["status"],
            start_time=last.get("startTime"),
            end_time=last.get("endTime"),
            partition=last.get("partition"),
        )
