import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative base so the built app works both at a domain root and under a
  // sub-path such as https://<user>.github.io/navier-stokes-sim/.
  base: './',
  build: {
    target: 'es2022',
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
