module.exports = {
  preset: "@vue/cli-plugin-unit-jest",
  transform: {
    "^.+\\.vue$": "vue-jest",
  },
  collectCoverageFrom: [
    "src/**/*.vue",
    "!src/**/*.spec.js",
    "!**/node_modules/**"
  ],
  coverageReporters: ["html", "text-summary"],
};
