import bundleAnalyzer from '@next/bundle-analyzer';

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  devIndicators: false,
  // Enable gzip compression for API responses
  compress: true,
  // Optimize package imports for better tree-shaking
  experimental: {
    optimizePackageImports: ['@mui/material', '@mui/icons-material', '@embedpdf/core'],
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

const withBundleAnalyzer = bundleAnalyzer({
  enabled: process.env.ANALYZE === 'true',
});

export default withBundleAnalyzer(nextConfig);
