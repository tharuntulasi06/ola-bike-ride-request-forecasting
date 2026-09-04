import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ola Bike Ride Demand Forecasting Dashboard",
  description:
    "Production operational control panel for spatiotemporal multi-step ride demand predictions in Chennai.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-darkBg text-slate-100 min-h-screen antialiased selection:bg-emerald-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
