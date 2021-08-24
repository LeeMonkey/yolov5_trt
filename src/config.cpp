#include <fstream>
#include <typeinfo>

#include <config.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/reader.h> 

template <typename T>
void parseKey(const rapidjson::Document& doc, 
		const std::string& key, 
		typename std::enable_if<std::is_arithmetic<T>::value, T>::type& value)
{
    if(!doc.HasMember(key.c_str()))
    {
        LOGERROR("Error, No Such Key: `%s`!", key.c_str());
        exit(-1);
    }

    if(typeid(T) == typeid(bool))
    {
        // typeid
        if(!doc[key.c_str()].IsBool())
        {
            LOGERROR("`%s` Type Should be bool!", key.c_str());
            exit(-1);
        }
        else
            value = doc[key.c_str()].GetBool();
    } else if (typeid(T) == typeid(int) || typeid(T) == typeid(size_t))
    {
        // typeid
        if(!doc[key.c_str()].IsInt())
        {
            LOGERROR("`%s` Type Should be int!", key.c_str());
            exit(-1);
        }
        else
            value = doc[key.c_str()].GetInt();
    }else if (typeid(T) == typeid(float) || typeid(value) == typeid(double))
    {
        // typeid
        if(!doc[key.c_str()].IsDouble())
        {
            LOGERROR("`%s` Type Should be float!", key.c_str());
            exit(-1);
        }
        else
            value = doc[key.c_str()].GetDouble();
    }
#if 0
    else if (typeid(T) == typeid(std::string)) 
    {
        // typeid
        if(!doc[key.c_str()].IsString())
        {
            LOGERROR("`%s` Type Should be string!", key.c_str());
            exit(-1);
        }
        else
            value = doc[key.c_str()].GetString();
    }
#endif
    else
    {
        LOGERROR("`%s` not match Type!", key.c_str());
        exit(-1);
    }
}

template <typename T>
void parseKey(const rapidjson::Document& doc, 
		const std::string& key, 
		typename std::enable_if<std::is_class<T>::value && std::is_same<T, std::string>::value, T>::type & value)
{
    if(!doc.HasMember(key.c_str()))
    {
        LOGERROR("Error, No Such Key: `%s`!", key.c_str());
        exit(-1);
    }

    // typeid
    if(!doc[key.c_str()].IsString())
    {
        LOGERROR("`%s` Type Should be string!", key.c_str());
        exit(-1);
    }
    else
        value = doc[key.c_str()].GetString();
}

template<class T>
void parseKey(const rapidjson::Document& doc, 
		const std::string& key, 
		std::vector<typename std::enable_if<std::is_arithmetic<T>::value, T>::type>& value)
{
    if(!doc.HasMember(key.c_str()))
    {
        LOGERROR("Error, No Such Key: `%s`!", key.c_str());
        exit(-1);
    }

    if(!doc[key.c_str()].IsArray())
    {
        LOGERROR("Error, `%s` not Array!", key.c_str());
        exit(-1);
    }

    for(rapidjson::SizeType i = 0; i < doc[key.c_str()].Size(); ++i)
    {
        if(typeid(T) == typeid(bool))
        {
            // typeid
            if(!doc[key.c_str()][i].IsBool())
            {
                LOGERROR("`%s` Type Should be bool!", key.c_str());
                exit(-1);
            }
            else
                value.push_back(doc[key.c_str()][i].GetBool());
        } else if (typeid(T) == typeid(int) || typeid(T) == typeid(size_t))
        {
            // typeid
            if(!doc[key.c_str()][i].IsInt())
            {
                LOGERROR("`%s` Type Should be int!", key.c_str());
                exit(-1);
            }
            else
                value.push_back(doc[key.c_str()][i].GetInt());
        }else if (typeid(T) == typeid(float) or typeid(T) == typeid(double))
        {
            // typeid
            if(!doc[key.c_str()][i].IsDouble())
            {
                LOGERROR("`%s` Type Should be float!", key.c_str());
                exit(-1);
            }
            else
                value.push_back(doc[key.c_str()][i].GetDouble());
        } 
#if 0
        else if (typeid(T) == typeid(std::string)) 
        {
            // typeid
            if(!doc[key.c_str()].IsString())
            {
                LOGERROR("`%s` Type Should be string!", key.c_str());
                exit(-1);
            }
            else
                value.push_back(doc[key.c_str()][i].GetString());
        }
#endif
        else
        {
            LOGERROR("`%s` not match Type!", key.c_str());
            exit(-1);
        }
    }
}

template<class T>
void parseKey(const rapidjson::Document& doc, 
		const std::string& key, 
		std::vector<typename std::enable_if<std::is_class<T>::value && std::is_same<T, std::string>::value, T>::type>& value)
{
    if(!doc.HasMember(key.c_str()))
    {
        LOGERROR("Error, No Such Key: `%s`!", key.c_str());
        exit(-1);
    }

    if(!doc[key.c_str()].IsArray())
    {
        LOGERROR("Error, `%s` not Array!", key.c_str());
        exit(-1);
    }

    for(rapidjson::SizeType i = 0; i < doc[key.c_str()].Size(); ++i)
    {
        // typeid
        if(!doc[key.c_str()][i].IsString())
        {
            LOGERROR("`%s` Type Should be string!", key.c_str());
            exit(-1);
        }
        else
            value.push_back(doc[key.c_str()][i].GetString());
    }
}

ConfigManager::ConfigManager()
{
    this->init(ENGINE_PATH);
}

void ConfigManager::init(const std::string& config_path)
{
    std::ifstream ifs(config_path);
    if(!ifs.is_open())
    {
        LOGERROR("Fail to Open Config File:\n\t%s", config_path.c_str());
        exit(-1);
    }

    rapidjson::Document doc;
    rapidjson::IStreamWrapper isw(ifs);

    if(doc.ParseStream(isw).HasParseError())
    {
        LOGERROR("Parse Json Error!");
        exit(-1);
    }

    parseKey<std::string>(doc, "EnginePath", this->enginePath);
    parseKey<std::string>(doc, "WeightsPath", this->weightsPath);
    parseKey<std::string>(doc, "InputBlobName", this->inputBlobName);
    parseKey<std::string>(doc, "OutputBlobName", this->outputBlobName);

    parseKey<int>(doc, "Width", this->width);
    parseKey<int>(doc, "Height", this->height);
    parseKey<int>(doc, "Channel", this->channel);

	parseKey<int>(doc, "NumAnchor", this->numAnchor);
	parseKey<size_t>(doc, "MaxOutputBBoxCount", this->maxOutputBBoxCount);
	parseKey<float>(doc, "DepthMultiple", this->depthMultiple);
	parseKey<float>(doc, "WidthMultiple", this->widthMultiple);

    parseKey<float>(doc, "NMSThresh", this->nmsThresh);
    parseKey<float>(doc, "ConfThresh", this->confThresh);
    parseKey<float>(doc, "IgnoreThresh", this->ignoreThresh);

    parseKey<std::string>(doc, "ClassName", this->className);
    parseKey<bool>(doc, "ReturnMultiBox", this->returnMultiBox);
    parseKey<bool>(doc, "IsGlobalNMS", this->isGlobalNMS);

	// show class
    parseKey<int>(doc, "ShowClassIds", this->showClassIds);
	
}
