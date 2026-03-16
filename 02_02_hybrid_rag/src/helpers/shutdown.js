/**
 * Graceful shutdown handler.
 */
export const onShutdown = (cleanup) => {
  let shuttingDown = false;
  
  const handler = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log("\n");
    await cleanup();
    process.exit(0);
  };

  process.on("SIGINT", handler);
  process.on("SIGTERM", handler);
  
  return handler;
};
