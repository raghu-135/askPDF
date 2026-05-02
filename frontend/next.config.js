/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
  // Enable gzip compression for API responses
  compress: true,
  // Proxy API requests to RAG service
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://rag-service:8000/api/:path*',
      },
    ];
  },
  // Optimize package imports for better tree-shaking
  experimental: {
    optimizePackageImports: ['@mui/material', '@mui/icons-material', '@mui/system', '@embedpdf/core'],
  },
  transpilePackages: [
    '@embedpdf/core',
    '@embedpdf/engines',
    '@embedpdf/models',
    '@embedpdf/utils',
    '@embedpdf/plugin-document-manager',
    '@embedpdf/plugin-viewport',
    '@embedpdf/plugin-scroll',
    '@embedpdf/plugin-render',
    '@embedpdf/plugin-zoom',
    '@embedpdf/plugin-interaction-manager',
    '@embedpdf/plugin-selection',
    '@embedpdf/plugin-annotation',
    '@embedpdf/plugin-history',
  ],
};

// Bundle analyzer configuration
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

module.exports = withBundleAnalyzer(nextConfig);
