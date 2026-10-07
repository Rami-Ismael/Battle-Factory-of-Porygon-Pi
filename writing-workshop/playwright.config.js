import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  testMatch: '**/*.spec.js',
  workers: 1,
  use: {baseURL:'http://127.0.0.1:5058', viewport:{width:1440,height:1000}},
  webServer: {command:'.venv/bin/python tests/serve.py',url:'http://127.0.0.1:5058',reuseExistingServer:false},
});
