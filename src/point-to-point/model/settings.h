#ifndef POINT_TO_POINT_MODEL_SETTINGS_H
#define POINT_TO_POINT_MODEL_SETTINGS_H

#include <cstdlib>
#include <string>

namespace ns3 {

inline const char *GetRunDirEnvName()
{
	return "PARALEON_RUN_DIR";
}

inline const char *GetDefaultRunDir()
{
	return "mix";
}

inline const char *GetSchemeEnvName()
{
	return "PARALEON_SCHEME";
}

inline const char *GetDefaultScheme()
{
	return "paraleon";
}

inline std::string GetRunDir()
{
	const char *run_dir = std::getenv(GetRunDirEnvName());
	if (run_dir != NULL && run_dir[0] != '\0')
		return std::string(run_dir);
	return std::string(GetDefaultRunDir());
}

inline std::string GetRunFile(const std::string &file_name)
{
	return GetRunDir() + "/" + file_name;
}

inline std::string GetScheme()
{
	const char *scheme = std::getenv(GetSchemeEnvName());
	if (scheme != NULL && scheme[0] != '\0')
		return std::string(scheme);
	return std::string(GetDefaultScheme());
}

inline bool IsScheme(const std::string &scheme)
{
	return GetScheme() == scheme;
}

} // namespace ns3

#endif