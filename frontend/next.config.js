/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
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

module.exports = nextConfig;
