"""
kv-extractor-mcp-server - Module Entry Point

This module serves as the entry point for running 'python -m kv-extractor-mcp-server' or when executed via the package entry point.
"""

import argparse
import sys
import logging
from .server import parse_args, setup_logging, server

def main():
    args = parse_args()
    # Initialize logger and get logger object
    specific_logger = setup_logging(args.log, args.logfile)
    # Also get the global logger with the same name if it exists (already initialized)
    logger = logging.getLogger("kv-extractor-mcp-server")
    
    if logger and logger.handlers:
        logger.info("MCP Server starting up from __main__.py...")
    server.run(show_banner=False)

if __name__ == "__main__":
    main()