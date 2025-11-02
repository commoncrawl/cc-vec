"""CC Stats handler for MCP server."""

import logging
from typing import Any, Dict, List

from mcp.types import TextContent
from .base import FilterHandler
from ... import stats as stats_function
from ..filter_utils import parse_filter_config_from_mcp

logger = logging.getLogger(__name__)


class CCStatsHandler(FilterHandler):
    """Handler for cc_stats MCP method."""

    def __init__(self, api_method=None):
        super().__init__(api_method)

    async def handle(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle cc_stats tool calls."""
        # Parse FilterConfig from MCP arguments
        filter_config = parse_filter_config_from_mcp(args)

        try:
            response = stats_function(filter_config)

            response_text = f"Statistics via {response.backend}:\n\n"

            if response.per_crawl_stats:
                # Build table
                response_text += f"Found statistics for {len(response.per_crawl_stats)} crawl(s):\n\n"
                response_text += f"{'Crawl ID':<20} {'Records':>15} {'Size (MB)':>12} {'Scanned (GB)':>14} {'Cost ($)':>12}\n"
                response_text += "-" * 85 + "\n"

                for stats in response.per_crawl_stats:
                    response_text += (
                        f"{stats.crawl_id:<20} "
                        f"{stats.estimated_records:>15,} "
                        f"{stats.estimated_size_mb:>12.2f} "
                        f"{stats.data_scanned_gb:>14.2f} "
                        f"{stats.estimated_cost_usd:>12.4f}\n"
                    )

                # Add totals if multiple crawls
                if len(response.per_crawl_stats) > 1:
                    response_text += "-" * 85 + "\n"
                    response_text += (
                        f"{'TOTAL':<20} "
                        f"{response.total_estimated_records:>15,} "
                        f"{response.total_estimated_size_mb:>12.2f} "
                        f"{response.total_data_scanned_gb:>14.2f} "
                        f"{response.total_estimated_cost_usd:>12.4f}\n"
                    )
            else:
                response_text += "No statistics found.\n"

            return [TextContent(type="text", text=response_text)]

        except Exception as e:
            error_text = f"Statistics calculation failed: {str(e)}"
            logger.error(error_text, exc_info=True)
            return [TextContent(type="text", text=error_text)]
