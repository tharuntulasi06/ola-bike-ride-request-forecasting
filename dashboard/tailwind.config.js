/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        monoBlack: "#000000",
        monoDark: "#0A0A0A",
        monoCard: "rgba(17, 17, 17, 0.75)",
        monoBorder: "#262626",
        monoHoverBorder: "#404040",
        monoText: "#FFFFFF",
        monoMuted: "#A1A1AA",
        monoSubtle: "#52525B",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
